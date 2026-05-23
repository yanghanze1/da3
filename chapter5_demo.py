"""
Chapter 5 Spatial Ranging and Height Estimation - Theory Visualization Demo
Based on thesis Chapter 5 theory, using sucai/1000016533.mp4

Chapter 5 Theory:
5.1 Measurement Baseline and Scale Alignment
5.2 Geometric Model and Coordinate Transformation
5.3.1 Maximum Sag (h_k)
5.3.2 Nearest Forward Distance (z_k)
5.3.3 Lateral Occupation Width (w_k)
5.3.4 Free Space at Same Depth (f_k)
5.4 Safety Thresholds and Risk Decision
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

DATASET_DIR = Path('d:/agent_lunwen/datasets/1000016533')
OUTPUT_DIR = Path('d:/agent_lunwen/outputs/chapter5_visualization')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_frames(frames_dir=DATASET_DIR / 'frames'):
    frame_paths = sorted(list(frames_dir.glob('frame_*.png')))
    frames = []
    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames

def simulate_road_mask(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    road_mask = np.zeros((h, w), dtype=np.uint8)
    horizon_line = int(h * 0.4)
    for y in range(h-1, horizon_line, -1):
        row_edges = edges[y, :]
        edge_density = np.mean(row_edges > 0)
        if edge_density < 0.15:
            road_mask[y, :] = 255
    kernel = np.ones((15, 15), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    road_mask[:horizon_line, :] = 0
    return road_mask, horizon_line

def simulate_depth_map(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    depth = np.ones((h, w)) * 10.0
    gradient_inv = 1.0 / (1.0 + gradient_mag * 0.1)
    depth = depth * (0.3 + gradient_inv * 0.7)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    perspective_factor = 1.0 + (y_coords / h) * 0.5
    depth = depth * perspective_factor
    noise = np.random.randn(h, w) * 0.3
    depth = np.clip(depth + noise, 0.5, 15.0)
    return depth.astype(np.float32)

def estimate_ground_plane(depth_map, road_mask):
    h, w = depth_map.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    valid_mask = road_mask > 0
    if np.sum(valid_mask) < 100:
        valid_mask = np.ones_like(road_mask, dtype=bool)
    fx, fy = w * 0.8, h * 0.8
    cx, cy = w // 2, h // 2
    X = (x_coords - cx) * depth_map / fx
    Y = (y_coords - cy) * depth_map / fy
    Z = depth_map
    valid_X = X[valid_mask]
    valid_Y = Y[valid_mask]
    valid_Z = Z[valid_mask]
    if len(valid_X) < 30:
        return lambda p: p[1] - 0.0, {'normal': [0, 1, 0], 'offset': 0.0, 'scale_factor': 1.0}
    points = np.column_stack([valid_X, valid_Z, np.ones_like(valid_X)])
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(points, valid_Y, rcond=None)
        a, b, c = coeffs
    except:
        a, b, c = 0, 0, 0
    def plane_func(point):
        X, Y, Z = point
        return a*X + b*Z + c - Y
    return plane_func, {'normal': [a, -1, b], 'offset': c, 'coeffs': [a, b, c], 'scale_factor': 1.0}

def scale_alignment(depth_map, plane_func, camera_height=0.4):
    """5.1 Scale Alignment - Transform relative depth to metric depth"""
    h, w = depth_map.shape
    scale_factor = camera_height / max(abs(plane_func([0, 0, 1.0])), 0.01)
    metric_depth = depth_map * scale_factor
    return metric_depth, scale_factor

def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """5.2 Transform pixel to camera 3D coordinates"""
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def camera_to_world_coords(cam_coords, pitch=0.0, yaw=0.0, camera_height=0.4):
    """5.2 Transform camera coords to world coords (simplified)"""
    X, Y, Z = cam_coords
    # World: X=lateral, Y=vertical, Z=forward
    # Camera tilted slightly
    Y_world = Y + camera_height
    Z_world = Z * np.cos(pitch) - Y * np.sin(pitch)
    return np.array([X, Y_world, Z_world])

def detect_anomaly_points(depth_map, road_mask, plane_func, threshold_m=-0.015):
    h, w = depth_map.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    fx, fy = w * 0.8, h * 0.8
    cx, cy = w // 2, h // 2
    X = (x_coords - cx) * depth_map / fx
    Y = (y_coords - cy) * depth_map / fy
    Z = depth_map
    signed_height = plane_func([X, Y, Z])
    anomaly_mask = np.zeros_like(road_mask)
    negative_mask = signed_height < threshold_m
    road_region = road_mask > 0
    anomaly_mask[road_region & negative_mask] = 255
    return anomaly_mask, signed_height

def form_candidate_clusters(anomaly_mask, signed_height, metric_depth_map):
    labeled_array, num_features = ndimage.label(anomaly_mask > 0)
    clusters = []
    h, w = anomaly_mask.shape
    fx, fy, cx, cy = w * 0.8, h * 0.8, w // 2, h // 2

    for cluster_id in range(1, num_features + 1):
        cluster_mask = (labeled_array == cluster_id)
        ys, xs = np.where(cluster_mask)
        if len(xs) < 3:
            continue

        heights = signed_height[cluster_mask]
        negative_ratio = np.mean(heights < -0.015)
        if negative_ratio < 0.2:
            continue

        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()

        # Compute 3D points in camera coords
        points_3d_cam = []
        for y, x in zip(ys, xs):
            depth = metric_depth_map[y, x]
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            points_3d_cam.append([X, Y, Z])

        points_3d_cam = np.array(points_3d_cam)

        clusters.append({
            'id': cluster_id,
            'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
            'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
            'area': len(xs),
            'negative_ratio': negative_ratio,
            'mean_height': np.mean(heights),
            'pixel_mask': cluster_mask,
            'points_3d_cam': points_3d_cam,
            'pixel_coords': list(zip(xs, ys))
        })
    return clusters

def compute_measurements(cluster, camera_height=0.4):
    """5.3 Compute geometric measurements for candidate cluster"""
    points_3d = cluster['points_3d_cam']

    if len(points_3d) == 0:
        return {'h_k': 0.0, 'z_k': 0.0, 'w_k': 0.0, 'f_k': 0.0, 'anchor': None}

    # Y in camera coords represents height relative to camera
    # Transform to world: Y_world = Y_cam + camera_height
    Y_world = points_3d[:, 1] + camera_height

    # World coords: X=lateral, Y=vertical, Z=forward
    X_world = points_3d[:, 0]
    Z_world = points_3d[:, 2]

    # 5.3.1 Maximum sag: h_k = -min(d_h) for negative candidates
    # d_h is negative for points below ground
    min_Y_world = np.min(Y_world)
    h_k = -min_Y_world  # Positive value for sag

    # 5.3.2 Nearest forward distance: z_k = min(Z)
    z_k = np.min(Z_world)

    # 5.3.3 Lateral occupation width: w_k = max(X) - min(X)
    w_k = np.max(X_world) - np.min(X_world)

    # 5.3.4 Free space at same depth (simplified estimation)
    # Estimate based on cluster extent
    f_k = w_k * 2.5  # Simplified: assume free space extends beyond cluster

    # Anchor point (closest to vehicle)
    anchor_idx = np.argmin(Z_world)
    anchor = points_3d[anchor_idx]

    return {
        'h_k': float(h_k),
        'z_k': float(z_k),
        'w_k': float(w_k),
        'f_k': float(f_k),
        'anchor': anchor,
        'X_world': X_world,
        'Y_world': Y_world,
        'Z_world': Z_world
    }

def classify_obstacle_type(cluster):
    """Classify as curb or pothole based on geometry"""
    min_x, min_y, w, h = cluster['bbox']
    boundary_touch = 0
    total_boundary = 0

    if min_x > 0:
        total_boundary += h
    if min_x + w < 1280 - 1:
        total_boundary += h
    if min_y > 0:
        total_boundary += w

    boundary_ratio = boundary_touch / max(total_boundary, 1)
    aspect_ratio = w / max(h, 1)
    is_elongated = aspect_ratio > 2.5 or aspect_ratio < 0.4

    if boundary_ratio > 0.3 and is_elongated:
        return 'curb'
    elif cluster['negative_ratio'] > 0.5 and cluster['area'] > 100:
        return 'pothole'
    elif boundary_ratio > 0.2:
        return 'curb'
    else:
        return 'pothole'

def risk_assessment(h_k, z_k, w_k, f_k, obstacle_type):
    """5.4 Safety Threshold and Risk Decision"""
    # Thresholds from thesis
    h_safe = 0.02   # Safe sag threshold (m)
    h_danger = 0.05 # Dangerous sag threshold (m)
    z_safe = 1.5    # Safe distance threshold (m)
    z_warning = 3.0 # Warning distance threshold (m)
    w_occupy = 0.65 # Vehicle width (m)

    # Compute occupation ratio
    occupy_ratio = w_k / max(f_k, 0.1)

    # Risk decision logic
    if h_k > h_danger and z_k < z_safe:
        decision = 'DANGER'
        level = 3
    elif h_k > h_safe and z_k < z_warning:
        if occupy_ratio > 0.5:
            decision = 'WARNING'
            level = 2
        else:
            decision = 'CAUTION'
            level = 1
    else:
        decision = 'SAFE'
        level = 0

    return {
        'decision': decision,
        'level': level,
        'h_k': h_k,
        'z_k': z_k,
        'w_k': w_k,
        'f_k': f_k,
        'occupy_ratio': occupy_ratio,
        'h_safe': h_safe,
        'h_danger': h_danger,
        'z_safe': z_safe,
        'z_warning': z_warning
    }

def visualize_chapter5(frames, output_dir):
    """Generate Chapter 5 visualization"""
    print("Generating Chapter 5 visualization...")

    selected_indices = [0, 30, 60, 90, 120]
    selected_indices = [i for i in selected_indices if i < len(frames)]
    if len(selected_indices) == 0:
        selected_indices = [0]

    all_results = []
    camera_height = 0.4  # meters

    for idx in selected_indices:
        frame = frames[idx]
        h, w = frame.shape[:2]
        print(f"Processing frame {idx}/{len(frames)}...")

        road_mask, horizon_line = simulate_road_mask(frame)
        depth_map = simulate_depth_map(frame)
        plane_func, plane_params = estimate_ground_plane(depth_map, road_mask)
        metric_depth, scale_factor = scale_alignment(depth_map, plane_func, camera_height)
        anomaly_mask, signed_height = detect_anomaly_points(depth_map, road_mask, plane_func)
        clusters = form_candidate_clusters(anomaly_mask, signed_height, metric_depth)

        # Compute measurements for each cluster
        frame_results = {
            'frame_idx': idx,
            'frame': frame,
            'road_mask': road_mask,
            'depth_map': depth_map,
            'metric_depth': metric_depth,
            'scale_factor': scale_factor,
            'anomaly_mask': anomaly_mask,
            'signed_height': signed_height,
            'clusters': []
        }

        for cluster in clusters:
            measurements = compute_measurements(cluster, camera_height)
            cluster['measurements'] = measurements
            cluster['obstacle_type'] = classify_obstacle_type(cluster)
            risk = risk_assessment(
                measurements['h_k'], measurements['z_k'],
                measurements['w_k'], measurements['f_k'],
                cluster['obstacle_type']
            )
            cluster['risk'] = risk
            frame_results['clusters'].append(cluster)

        all_results.append(frame_results)

    n_results = len(all_results)

    # ===== FIGURE 1: 5.1 Scale Alignment =====
    fig1, axes1 = plt.subplots(min(3, n_results), 3, figsize=(15, 5*min(3, n_results)))
    if min(3, n_results) == 1:
        axes1 = axes1.reshape(1, -1)
    fig1.suptitle('5.1 Scale Alignment - Relative to Metric Depth\n(D_t = s * D_rel)', fontsize=14, fontweight='bold')

    for i, res in enumerate(all_results[:min(3, n_results)]):
        frame = res['frame']
        rel_depth = res['depth_map']
        metric_depth = res['metric_depth']
        scale = res['scale_factor']

        axes1[i, 0].imshow(rel_depth, cmap='viridis')
        axes1[i, 0].set_title(f'Frame {res["frame_idx"]}: Relative Depth D_rel\n(No scale)', fontsize=10)
        axes1[i, 0].axis('off')
        plt.colorbar(axes1[i, 0].imshow(rel_depth, cmap='viridis'), ax=axes1[i, 0], fraction=0.046)

        axes1[i, 1].imshow(metric_depth, cmap='plasma')
        axes1[i, 1].set_title(f'Metric Depth D_t\nScale factor s={scale:.3f}', fontsize=10)
        axes1[i, 1].axis('off')
        plt.colorbar(axes1[i, 1].imshow(metric_depth, cmap='plasma'), ax=axes1[i, 1], fraction=0.046)

        # Show scale info
        axes1[i, 2].text(0.5, 0.7, f'Scale Alignment', fontsize=12, ha='center', fontweight='bold',
                         transform=axes1[i, 2].transAxes)
        axes1[i, 2].text(0.5, 0.5, f'Camera height: {camera_height:.2f} m\n'
                         f'Virtual camera height: {plane_params.get("offset", 0):.3f}\n'
                         f'Scale factor: {scale:.4f}', fontsize=10, ha='center',
                         transform=axes1[i, 2].transAxes)
        axes1[i, 2].text(0.5, 0.2, 'D_t = (h_cam / h_cam_est) * D_rel', fontsize=9, ha='center',
                         transform=axes1[i, 2].transAxes, family='monospace')
        axes1[i, 2].axis('off')

    plt.tight_layout()
    fig1.savefig(output_dir / 'fig5_1_scale_alignment.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig5_1_scale_alignment.png")

    # ===== FIGURE 2: 5.2 Geometric Model =====
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('5.2 Geometric Model and Coordinate Transformation', fontsize=14, fontweight='bold')

    # Coordinate system illustration
    axes2[0].text(0.5, 0.9, 'World Coordinate System', fontsize=12, ha='center', fontweight='bold',
                  transform=axes2[0].transAxes)
    axes2[0].text(0.5, 0.7, 'X: Lateral position\nY: Vertical height\nZ: Forward distance', fontsize=11,
                  ha='center', va='center', transform=axes2[0].transAxes)
    axes2[0].annotate('', xy=(0.7, 0.5), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle='->', color='red', lw=2))
    axes2[0].annotate('X', xy=(0.72, 0.5), fontsize=10, color='red')
    axes2[0].annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle='->', color='green', lw=2))
    axes2[0].annotate('Y', xy=(0.5, 0.72), fontsize=10, color='green')
    axes2[0].annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    axes2[0].annotate('Z', xy=(0.5, 0.28), fontsize=10, color='blue')
    axes2[0].set_xlim(0, 1)
    axes2[0].set_ylim(0, 1)
    axes2[0].axis('off')

    # Show 3D point cloud for first cluster
    if all_results and all_results[0]['clusters']:
        cluster = all_results[0]['clusters'][0]
        meas = cluster['measurements']
        if 'X_world' in meas:
            ax3d = fig2.add_subplot(1, 3, 3, projection='3d')
            ax3d.scatter(meas['X_world'][::10], meas['Z_world'][::10], meas['Y_world'][::10],
                        c='red', s=5, alpha=0.6)
            ax3d.set_xlabel('X (m)')
            ax3d.set_ylabel('Z (m)')
            ax3d.set_zlabel('Y (m)')
            ax3d.set_title('3D Point Cloud (Sample)')
            # Remove the text axes from subplot position
            axes2[2].set_position([0.7, 0.1, 0.25, 0.8])
        else:
            axes2[2].text(0.5, 0.5, 'No 3D points available', ha='center', va='center')
            axes2[2].axis('off')
    else:
        axes2[2].text(0.5, 0.5, 'No clusters', ha='center', va='center')
        axes2[2].axis('off')

    # Anchor point explanation
    axes2[1].text(0.5, 0.9, 'Anchor Point (Eq.31)', fontsize=12, ha='center', fontweight='bold',
                  transform=axes2[1].transAxes)
    axes2[1].text(0.5, 0.7, 'P_anchor = argmin ||P - P_vehicle||\n\n'
                 'The closest point on candidate\nto the vehicle position.\n\n'
                 'Used for nearest distance\nmeasurement z_k.', fontsize=10, ha='center',
                 transform=axes2[1].transAxes)
    axes2[1].axis('off')

    plt.tight_layout()
    fig2.savefig(output_dir / 'fig5_2_geometric_model.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig5_2_geometric_model.png")

    # ===== FIGURE 3: 5.3 Measurements =====
    fig3, axes3 = plt.subplots(min(3, n_results), 4, figsize=(20, 5*min(3, n_results)))
    if min(3, n_results) == 1:
        axes3 = axes3.reshape(1, -1)
    fig3.suptitle('5.3 Geometric Measurements\n(h_k, z_k, w_k, f_k)', fontsize=14, fontweight='bold')

    for i, res in enumerate(all_results[:min(3, n_results)]):
        frame = res['frame']
        clusters = res['clusters']

        # Show frame with measurements
        display_frame = frame.copy()
        for cluster in clusters:
            min_x, min_y, bw, bh = cluster['bbox']
            meas = cluster['measurements']
            obstacle_type = cluster['obstacle_type']

            color = (1.0, 0.65, 0.0) if obstacle_type == 'curb' else (0.54, 0.17, 0.89)

            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=2,
                                    edgecolor=color, facecolor=color, alpha=0.3)
            axes3[i, 0].add_patch(rect)

            # Label with measurements
            label = f"h={meas['h_k']:.2f}m z={meas['z_k']:.1f}m"
            axes3[i, 0].text(min_x, min_y-10, label, fontsize=7, color='white',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        axes3[i, 0].imshow(display_frame)
        axes3[i, 0].set_title(f'Frame {res["frame_idx"]}: Candidates with Measurements', fontsize=10)
        axes3[i, 0].axis('off')

        # Measurement bar chart
        if clusters:
            h_vals = [c['measurements']['h_k'] for c in clusters[:5]]
            z_vals = [c['measurements']['z_k'] for c in clusters[:5]]
            w_vals = [c['measurements']['w_k'] for c in clusters[:5]]
            ids = [f'C{c["id"]}' for c in clusters[:5]]

            x = np.arange(len(ids))
            width = 0.25

            axes3[i, 1].bar(x - width, h_vals, width, label='h_k (m)', color='red', alpha=0.7)
            axes3[i, 1].bar(x, z_vals, width, label='z_k (m)', color='green', alpha=0.7)
            axes3[i, 1].bar(x + width, w_vals, width, label='w_k (m)', color='blue', alpha=0.7)
            axes3[i, 1].set_xticks(x)
            axes3[i, 1].set_xticklabels(ids)
            axes3[i, 1].set_ylabel('Distance (m)')
            axes3[i, 1].set_title('Measurements (h_k, z_k, w_k)')
            axes3[i, 1].legend(fontsize=8)
        else:
            axes3[i, 1].text(0.5, 0.5, 'No clusters', ha='center', va='center')
            axes3[i, 1].axis('off')

        # Depth profile
        if clusters and 'Z_world' in clusters[0]['measurements']:
            axes3[i, 2].clear()
            for j, cluster in enumerate(clusters[:3]):
                meas = cluster['measurements']
                Z = meas['Z_world']
                Y = meas['Y_world']
                axes3[i, 2].scatter(Z[::5], Y[::5], s=3, alpha=0.5, label=f"C{cluster['id']}")
            axes3[i, 2].axhline(y=0, color='k', linestyle='--', linewidth=1, label='Ground')
            axes3[i, 2].set_xlabel('Z (Forward, m)')
            axes3[i, 2].set_ylabel('Y (Height, m)')
            axes3[i, 2].set_title('Depth-Height Profile')
            axes3[i, 2].legend(fontsize=8)
        else:
            axes3[i, 2].text(0.5, 0.5, 'No profile data', ha='center', va='center')
            axes3[i, 2].axis('off')

        # Measurement table
        axes3[i, 3].axis('off')
        if clusters:
            table_data = []
            for c in clusters[:5]:
                m = c['measurements']
                r = c['risk']
                table_data.append([
                    f"C{c['id']}",
                    f"{m['h_k']:.3f}",
                    f"{m['z_k']:.2f}",
                    f"{m['w_k']:.3f}",
                    r['decision'][:4]
                ])
            table = axes3[i, 3].table(
                cellText=table_data,
                colLabels=['ID', 'h_k(m)', 'z_k(m)', 'w_k(m)', 'Risk'],
                loc='center', cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.2)
            axes3[i, 3].set_title('Measurement Summary', fontsize=10)
        else:
            axes3[i, 3].text(0.5, 0.5, 'No data', ha='center')

    plt.tight_layout()
    fig3.savefig(output_dir / 'fig5_3_measurements.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig5_3_measurements.png")

    # ===== FIGURE 4: 5.4 Risk Assessment =====
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
    fig4.suptitle('5.4 Safety Thresholds and Risk Decision\n(Rule-based Risk Classification)', fontsize=14, fontweight='bold')

    # Risk decision flow
    axes4[0, 0].text(0.5, 0.9, 'Risk Decision Flow', fontsize=12, ha='center', fontweight='bold',
                    transform=axes4[0, 0].transAxes)
    flow_text = """
    Input: h_k, z_k, w_k, f_k

    Check h_k > h_danger (0.05m) AND z_k < z_safe (1.5m)?
      YES -> DANGER (Level 3)
      NO  -> Check h_k > h_safe (0.02m) AND z_k < z_warning (3.0m)?
               YES -> Check occupy_ratio > 50%?
                        YES -> WARNING (Level 2)
                        NO  -> CAUTION (Level 1)
               NO  -> SAFE (Level 0)
    """
    axes4[0, 0].text(0.1, 0.7, flow_text, fontsize=9, transform=axes4[0, 0].transAxes,
                     family='monospace', verticalalignment='top')
    axes4[0, 0].axis('off')

    # Risk distribution
    risk_counts = {'SAFE': 0, 'CAUTION': 0, 'WARNING': 0, 'DANGER': 0}
    for res in all_results:
        for c in res['clusters']:
            risk_counts[c['risk']['decision']] += 1

    colors_risk = ['green', 'yellow', 'orange', 'red']
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())

    axes4[0, 1].bar(labels, values, color=colors_risk, alpha=0.7)
    axes4[0, 1].set_ylabel('Count')
    axes4[0, 1].set_title('Risk Level Distribution')
    for j, v in enumerate(values):
        axes4[0, 1].text(j, v + 0.1, str(v), ha='center', fontsize=11)

    # Risk decision visualization on frame
    if all_results:
        res = all_results[0]
        frame = res['frame']
        clusters = res['clusters']

        display_frame = frame.copy()
        for cluster in clusters:
            min_x, min_y, bw, bh = cluster['bbox']
            risk = cluster['risk']
            meas = cluster['measurements']

            if risk['decision'] == 'DANGER':
                color = (1.0, 0.0, 0.0)
            elif risk['decision'] == 'WARNING':
                color = (1.0, 0.65, 0.0)
            elif risk['decision'] == 'CAUTION':
                color = (1.0, 1.0, 0.0)
            else:
                color = (0.0, 1.0, 0.0)

            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=3,
                                    edgecolor=color, facecolor=color, alpha=0.4)
            axes4[0, 2].add_patch(rect)
            text_color = 'white' if risk['decision'] != 'SAFE' else 'black'
            axes4[0, 2].text(min_x, min_y-10, risk['decision'], fontsize=9, color=text_color,
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

        axes4[0, 2].imshow(display_frame)
        axes4[0, 2].set_title(f'Frame {res["frame_idx"]}: Risk Visualization')
        axes4[0, 2].axis('off')
    else:
        axes4[0, 2].text(0.5, 0.5, 'No data', ha='center')
        axes4[0, 2].axis('off')

    # Threshold parameters
    axes4[1, 0].text(0.5, 0.9, 'Safety Thresholds (Table 5-2)', fontsize=12, ha='center', fontweight='bold',
                    transform=axes4[1, 0].transAxes)
    thresh_data = [
        ['Parameter', 'Safe', 'Warning', 'Danger'],
        ['Height h_k', '< 0.02m', '0.02-0.05m', '> 0.05m'],
        ['Distance z_k', '> 3.0m', '1.5-3.0m', '< 1.5m'],
        ['Occupy Ratio', '< 30%', '30-50%', '> 50%'],
    ]
    table = axes4[1, 0].table(cellText=thresh_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for j in range(4):
        table[(0, j)].set_facecolor('#cccccc')
    axes4[1, 0].axis('off')

    # Risk vs Distance scatter
    axes4[1, 1].clear()
    for res in all_results:
        for c in res['clusters']:
            meas = c['measurements']
            risk = c['risk']
            color = {'SAFE': 'green', 'CAUTION': 'yellow', 'WARNING': 'orange', 'DANGER': 'red'}[risk['decision']]
            axes4[1, 1].scatter(meas['z_k'], meas['h_k'], c=color, s=50, alpha=0.7)
    axes4[1, 1].axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='Danger threshold')
    axes4[1, 1].axhline(y=0.02, color='orange', linestyle='--', linewidth=1, label='Warning threshold')
    axes4[1, 1].axvline(x=1.5, color='red', linestyle=':', linewidth=1)
    axes4[1, 1].axvline(x=3.0, color='orange', linestyle=':', linewidth=1)
    axes4[1, 1].set_xlabel('Distance z_k (m)')
    axes4[1, 1].set_ylabel('Height h_k (m)')
    axes4[1, 1].set_title('Risk: Height vs Distance')
    axes4[1, 1].legend(fontsize=8)
    axes4[1, 1].set_xlim(0, 10)
    axes4[1, 1].set_ylim(0, 0.3)

    # Stopping distance formula
    axes4[1, 2].text(0.5, 0.9, 'Stopping Distance (Eq.38)', fontsize=12, ha='center', fontweight='bold',
                    transform=axes4[1, 2].transAxes)
    axes4[1, 2].text(0.5, 0.7, 'd_stop = v * t + v^2 / (2 * a)\n\n'
                     'Where:\n'
                     '  v = current speed (m/s)\n'
                     '  t = reaction time (s)\n'
                     '  a = deceleration (m/s^2)\n\n'
                     'Example:\n'
                     '  v = 1.0 m/s, t = 0.5 s, a = 1.2 m/s^2\n'
                     '  d_stop = 1.0*0.5 + 1.0^2/(2*1.2)\n'
                     '         = 0.5 + 0.42 = 0.92 m', fontsize=10, ha='center',
                     transform=axes4[1, 2].transAxes, family='monospace')
    axes4[1, 2].axis('off')

    plt.tight_layout()
    fig4.savefig(output_dir / 'fig5_4_risk_assessment.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig5_4_risk_assessment.png")

    # ===== FIGURE 5: Complete Chapter 5 Pipeline =====
    fig5, axes5 = plt.subplots(2, 4, figsize=(20, 10))
    fig5.suptitle('Chapter 5 Complete Pipeline\n(Spatial Ranging and Risk Assessment)', fontsize=16, fontweight='bold')

    if all_results:
        res = all_results[0]
        frame = res['frame']
        clusters = res['clusters']

        # Step 1: Input
        axes5[0, 0].imshow(frame)
        axes5[0, 0].set_title('(1) Input Image', fontsize=10)
        axes5[0, 0].axis('off')

        # Step 2: Metric Depth
        im = axes5[0, 1].imshow(res['metric_depth'], cmap='plasma')
        axes5[0, 1].set_title('(2) Metric Depth D_t', fontsize=10)
        axes5[0, 1].axis('off')
        plt.colorbar(im, ax=axes5[0, 1], fraction=0.046)

        # Step 3: Candidates
        for c in clusters:
            min_x, min_y, bw, bh = c['bbox']
            color = (1.0, 0.65, 0.0) if c['obstacle_type'] == 'curb' else (0.54, 0.17, 0.89)
            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=2,
                                    edgecolor=color, facecolor=color, alpha=0.3)
            axes5[0, 2].add_patch(rect)
        axes5[0, 2].imshow(frame)
        axes5[0, 2].set_title(f'(3) Candidates ({len(clusters)})', fontsize=10)
        axes5[0, 2].axis('off')

        # Step 4: Measurements
        axes5[0, 3].bar(range(min(5, len(clusters))),
                        [c['measurements']['h_k'] for c in clusters[:5]], color='red', alpha=0.7)
        axes5[0, 3].set_title('(4) Height h_k', fontsize=10)
        axes5[0, 3].set_ylabel('meters')
        axes5[0, 3].set_xticks(range(min(5, len(clusters))))
        axes5[0, 3].set_xticklabels([f'C{c["id"]}' for c in clusters[:5]], fontsize=8)

        # Step 5: Distance
        axes5[1, 0].bar(range(min(5, len(clusters))),
                        [c['measurements']['z_k'] for c in clusters[:5]], color='green', alpha=0.7)
        axes5[1, 0].set_title('(5) Distance z_k', fontsize=10)
        axes5[1, 0].set_ylabel('meters')
        axes5[1, 0].set_xticks(range(min(5, len(clusters))))
        axes5[1, 0].set_xticklabels([f'C{c["id"]}' for c in clusters[:5]], fontsize=8)

        # Step 6: Width
        axes5[1, 1].bar(range(min(5, len(clusters))),
                        [c['measurements']['w_k'] for c in clusters[:5]], color='blue', alpha=0.7)
        axes5[1, 1].set_title('(6) Width w_k', fontsize=10)
        axes5[1, 1].set_ylabel('meters')
        axes5[1, 1].set_xticks(range(min(5, len(clusters))))
        axes5[1, 1].set_xticklabels([f'C{c["id"]}' for c in clusters[:5]], fontsize=8)

        # Step 7: Risk levels
        risk_labels = [c['risk']['decision'] for c in clusters[:5]]
        risk_colors = [{'SAFE': 'green', 'CAUTION': 'yellow', 'WARNING': 'orange', 'DANGER': 'red'}[r] for r in risk_labels]
        axes5[1, 2].bar(range(min(5, len(clusters))), [1]*min(5, len(clusters)), color=risk_colors, alpha=0.7)
        axes5[1, 2].set_title('(7) Risk Levels', fontsize=10)
        axes5[1, 2].set_xticks(range(min(5, len(clusters))))
        axes5[1, 2].set_xticklabels([f'C{c["id"]}' for c in clusters[:5]], fontsize=8)
        axes5[1, 2].set_yticks([])

        # Step 8: Output
        axes5[1, 3].text(0.5, 0.9, 'Risk Summary', fontsize=12, ha='center', fontweight='bold',
                         transform=axes5[1, 3].transAxes)
        safe_n = sum(1 for c in clusters if c['risk']['decision'] == 'SAFE')
        caution_n = sum(1 for c in clusters if c['risk']['decision'] == 'CAUTION')
        warning_n = sum(1 for c in clusters if c['risk']['decision'] == 'WARNING')
        danger_n = sum(1 for c in clusters if c['risk']['decision'] == 'DANGER')
        axes5[1, 3].text(0.5, 0.7, f'SAFE: {safe_n}\nCAUTION: {caution_n}\nWARNING: {warning_n}\nDANGER: {danger_n}',
                        fontsize=11, ha='center', transform=axes5[1, 3].transAxes,
                        family='monospace')
        axes5[1, 3].axis('off')

    plt.tight_layout()
    fig5.savefig(output_dir / 'fig5_complete_pipeline.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig5_complete_pipeline.png")

    # Generate summary report
    generate_summary_report(all_results, output_dir)
    print(f"\nAll results saved to: {output_dir}")
    return all_results

def generate_summary_report(results, output_dir):
    """Generate Chapter 5 summary report"""
    report = []
    report.append("=" * 70)
    report.append("Chapter 5 Spatial Ranging and Height Estimation - Experiment Summary")
    report.append("=" * 70)
    report.append("")
    report.append("## 5.1 Scale Alignment")
    report.append("- Transform relative depth D_rel to metric depth D_t")
    report.append("- D_t = (camera_height / virtual_height) * D_rel")
    report.append("- Enables physical measurements in meters")
    report.append("")
    report.append("## 5.2 Geometric Model")
    report.append("- World coordinate: X=lateral, Y=vertical, Z=forward")
    report.append("- Anchor point P_anchor = argmin ||P - P_vehicle||")
    report.append("- All measurements based on world coordinates")
    report.append("")
    report.append("## 5.3 Geometric Measurements")
    report.append("- h_k: Maximum sag = -min(Y_world) (Eq.33)")
    report.append("- z_k: Nearest forward distance = min(Z_world) (Eq.34)")
    report.append("- w_k: Lateral width = max(X) - min(X) (Eq.35)")
    report.append("- f_k: Free space at same depth (Eq.36)")
    report.append("")
    report.append("## 5.4 Safety Thresholds")
    report.append("- h_safe = 0.02m, h_danger = 0.05m")
    report.append("- z_safe = 1.5m, z_warning = 3.0m")
    report.append("- Risk = f(h_k, z_k, occupy_ratio)")
    report.append("")
    report.append("## Experiment Results")
    report.append("")
    report.append("Frame | Clusters | DANGER | WARNING | CAUTION | SAFE")
    report.append("------ | -------- | ------ | ------- | ------- | ----")

    for res in results:
        total = len(res['clusters'])
        danger_n = sum(1 for c in res['clusters'] if c['risk']['decision'] == 'DANGER')
        warning_n = sum(1 for c in res['clusters'] if c['risk']['decision'] == 'WARNING')
        caution_n = sum(1 for c in res['clusters'] if c['risk']['decision'] == 'CAUTION')
        safe_n = sum(1 for c in res['clusters'] if c['risk']['decision'] == 'SAFE')

        report.append(f"  {res['frame_idx']}   |    {total}    |   {danger_n}   |    {warning_n}    |    {caution_n}    |   {safe_n}")

    report.append("")
    report.append("## Measurement Statistics")
    all_h = []
    all_z = []
    all_w = []
    for res in results:
        for c in res['clusters']:
            m = c['measurements']
            all_h.append(m['h_k'])
            all_z.append(m['z_k'])
            all_w.append(m['w_k'])

    if all_h:
        report.append(f"- Height h_k: min={min(all_h):.4f}, max={max(all_h):.4f}, mean={np.mean(all_h):.4f} m")
        report.append(f"- Distance z_k: min={min(all_z):.4f}, max={max(all_z):.4f}, mean={np.mean(all_z):.4f} m")
        report.append(f"- Width w_k: min={min(all_w):.4f}, max={max(all_w):.4f}, mean={np.mean(all_w):.4f} m")

    report_path = output_dir / 'chapter5_summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"Saved: chapter5_summary_report.txt")

def main():
    print("=" * 70)
    print("Chapter 5 Spatial Ranging - Visualization Demo")
    print("=" * 70)

    print("\n[1/3] Loading video frames...")
    frames = load_frames()
    print(f"Loaded {len(frames)} frames")

    if len(frames) == 0:
        print("Error: Cannot load frames")
        return

    print("\n[2/3] Generating Chapter 5 visualizations...")
    results = visualize_chapter5(frames, OUTPUT_DIR)

    print("\n[3/3] Done!")
    return results

if __name__ == '__main__':
    main()
"""
Chapter 4 Road Obstacle Detection - Theory Visualization Demo
Based on thesis Chapter 4 theory, using sucai/1000016533.mp4
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

# Use sans-serif font (English only for compatibility)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# Paths
DATASET_DIR = Path('d:/agent_lunwen/datasets/1000016533')
OUTPUT_DIR = Path('d:/agent_lunwen/outputs/chapter4_visualization')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_frames(frames_dir=DATASET_DIR / 'frames'):
    """Load video frames"""
    frame_paths = sorted(list(frames_dir.glob('frame_*.png')))
    frames = []
    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames

def simulate_road_mask(frame):
    """Simulate road mask (PIDNet-S like output)"""
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
    """Simulate depth map (Depth Anything 3 like output)"""
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
    """Estimate ground plane using RANSAC"""
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
        return lambda p: p[1] - 0.0, {'normal': [0, 1, 0], 'offset': 0.0}
    points = np.column_stack([valid_X, valid_Z, np.ones_like(valid_X)])
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(points, valid_Y, rcond=None)
        a, b, c = coeffs
    except:
        a, b, c = 0, 0, 0
    def plane_func(point):
        X, Y, Z = point
        return a*X + b*Z + c - Y
    return plane_func, {'normal': [a, -1, b], 'offset': c, 'coeffs': [a, b, c]}

def detect_anomaly_points(depth_map, road_mask, plane_func, threshold_m=-0.015):
    """Point-level anomaly detection"""
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

def form_candidate_clusters(anomaly_mask, signed_height):
    """Cluster-level candidate formation"""
    labeled_array, num_features = ndimage.label(anomaly_mask > 0)
    clusters = []
    h, w = anomaly_mask.shape
    for cluster_id in range(1, num_features + 1):
        cluster_mask = (labeled_array == cluster_id)
        ys, xs = np.where(cluster_mask)
        if len(xs) < 2:
            continue
        cluster_points_3d = []
        fx, fy = w * 0.8, h * 0.8
        cx, cy = w // 2, h // 2
        for y, x in zip(ys, xs):
            X = (x - cx) * signed_height[y, x] * 0.3 / fx
            Y = (y - cy) * signed_height[y, x] * 0.3 / fy
            Z = signed_height[y, x]
            cluster_points_3d.append([X, Y, Z])
        cluster_points_3d = np.array(cluster_points_3d)
        heights = signed_height[cluster_mask]
        negative_ratio = np.mean(heights < -0.015)
        if negative_ratio < 0.2:
            continue
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        clusters.append({
            'id': cluster_id,
            'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
            'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
            'area': len(xs),
            'negative_ratio': negative_ratio,
            'points': cluster_points_3d,
            'pixel_mask': cluster_mask,
            'mean_height': np.mean(heights)
        })
    return clusters

def classify_candidate_type(cluster, road_mask):
    """Candidate-level obstacle type classification"""
    min_x, min_y, w, h = cluster['bbox']
    boundary_touch = 0
    total_boundary = 0
    if min_x > 0:
        left_edge = road_mask[min_y:min_y+h, min_x-1]
        boundary_touch += np.sum(left_edge > 0)
        total_boundary += h
    if min_x + w < road_mask.shape[1] - 1:
        right_edge = road_mask[min_y:min_y+h, min_x+w+1]
        boundary_touch += np.sum(right_edge > 0)
        total_boundary += h
    if min_y > 0:
        top_edge = road_mask[min_y-1, min_x:min_x+w]
        boundary_touch += np.sum(top_edge > 0)
        total_boundary += w
    boundary_ratio = boundary_touch / max(total_boundary, 1)
    aspect_ratio = w / max(h, 1)
    is_elongated = aspect_ratio > 2.5 or aspect_ratio < 0.4
    if boundary_ratio > 0.3 and is_elongated:
        obstacle_type = 'curb'
    elif cluster['negative_ratio'] > 0.5 and cluster['area'] > 100:
        obstacle_type = 'pothole'
    elif boundary_ratio > 0.2:
        obstacle_type = 'curb'
    else:
        obstacle_type = 'pothole'
    return obstacle_type, {'boundary_ratio': boundary_ratio, 'aspect_ratio': aspect_ratio}

def visualize_chapter4(frames, output_dir):
    """Generate Chapter 4 visualization results"""
    print("Generating Chapter 4 visualization...")

    selected_indices = [0, 30, 60, 90, 120]
    selected_indices = [i for i in selected_indices if i < len(frames)]
    if len(selected_indices) == 0:
        selected_indices = [0]

    results = []

    for idx in selected_indices:
        frame = frames[idx]
        h, w = frame.shape[:2]
        print(f"Processing frame {idx}/{len(frames)}...")

        road_mask, horizon_line = simulate_road_mask(frame)
        depth_map = simulate_depth_map(frame)
        plane_func, plane_params = estimate_ground_plane(depth_map, road_mask)
        anomaly_mask, signed_height = detect_anomaly_points(depth_map, road_mask, plane_func)
        clusters = form_candidate_clusters(anomaly_mask, signed_height)

        for cluster in clusters:
            obstacle_type, type_info = classify_candidate_type(cluster, road_mask)
            cluster['obstacle_type'] = obstacle_type
            cluster['type_info'] = type_info

        results.append({
            'frame_idx': idx,
            'frame': frame,
            'road_mask': road_mask,
            'depth_map': depth_map,
            'anomaly_mask': anomaly_mask,
            'signed_height': signed_height,
            'clusters': clusters,
            'horizon_line': horizon_line
        })

    # ====== FIGURE 1: 4.1 Obstacle Type Definition ======
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('4.1 Obstacle Type Definition\n(Positive vs Negative Obstacles)', fontsize=14, fontweight='bold')

    axes1[0].imshow(frame)
    axes1[0].set_title('Positive Obstacle\nAbove ground - Collision risk', fontsize=10)
    axes1[0].axhline(y=h*0.6, color='green', linestyle='--', linewidth=2)
    axes1[0].add_patch(patches.Rectangle((w*0.4, h*0.35), w*0.15, h*0.25,
                                          linewidth=3, edgecolor='red', facecolor='red', alpha=0.5))
    axes1[0].annotate('Obstacle', xy=(w*0.47, h*0.3), fontsize=10, color='red', ha='center')
    axes1[0].annotate('Collision', xy=(w*0.47, h*0.65), fontsize=9, color='red', ha='center')
    axes1[0].axis('off')

    axes1[1].imshow(frame)
    axes1[1].set_title('Negative Obstacle\nBelow ground - Fall risk', fontsize=10)
    axes1[1].axhline(y=h*0.6, color='green', linestyle='--', linewidth=2)
    axes1[1].add_patch(patches.Rectangle((w*0.3, h*0.6), w*0.35, h*0.05,
                                          linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3))
    axes1[1].add_patch(patches.Rectangle((w*0.45, h*0.65), w*0.12, h*0.08,
                                          linewidth=2, edgecolor='darkblue', facecolor='darkblue', alpha=0.5))
    axes1[1].annotate('Pothole', xy=(w*0.51, h*0.7), fontsize=10, color='darkblue', ha='center')
    axes1[1].annotate('Curb', xy=(w*0.3, h*0.58), fontsize=9, color='blue', ha='center')
    axes1[1].axis('off')

    axes1[2].text(0.5, 0.8, 'Obstacle Set', fontsize=12, ha='center', fontweight='bold',
                  transform=axes1[2].transAxes)
    axes1[2].text(0.3, 0.55, 'Positive\n- Pedestrian\n- Vehicle\n- Box', fontsize=10, ha='center',
                  transform=axes1[2].transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    axes1[2].text(0.7, 0.55, 'Negative\n- Curb drop\n- Pothole\n- Depression', fontsize=10, ha='center',
                  transform=axes1[2].transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes1[2].annotate('', xy=(0.42, 0.65), xytext=(0.35, 0.72), arrowprops=dict(arrowstyle='->', color='gray'))
    axes1[2].annotate('', xy=(0.58, 0.65), xytext=(0.65, 0.72), arrowprops=dict(arrowstyle='->', color='gray'))
    axes1[2].set_xlim(0, 1)
    axes1[2].set_ylim(0, 1)
    axes1[2].axis('off')

    plt.tight_layout()
    fig1.savefig(output_dir / 'fig4_1_obstacle_type_definition.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig4_1_obstacle_type_definition.png")

    # ====== FIGURE 2: 4.2 Image Preprocessing ======
    n_results = len(results)
    fig2, axes2 = plt.subplots(min(3, n_results), 3, figsize=(15, 5*min(3, n_results)))
    if min(3, n_results) == 1:
        axes2 = axes2.reshape(1, -1)
    fig2.suptitle('4.2 Image Preprocessing - Road Mask Generation\n(PIDNet-S Output)', fontsize=14, fontweight='bold')

    for i, res in enumerate(results[:3]):
        if i >= min(3, n_results):
            break
        frame = res['frame']
        road_mask = res['road_mask']

        axes2[i, 0].imshow(frame)
        axes2[i, 0].set_title(f'Frame {res["frame_idx"]}: Input Image It', fontsize=10)
        axes2[i, 0].axhline(y=res['horizon_line'], color='yellow', linestyle='--', linewidth=1.5)
        axes2[i, 0].axis('off')

        road_display = np.zeros_like(frame)
        road_display[road_mask > 0] = [0, 255, 0]
        road_display[road_mask == 0] = frame[road_mask == 0] * 0.5
        axes2[i, 1].imshow(road_display)
        axes2[i, 1].set_title('Road Mask Mt\n(PIDNet-S)', fontsize=10)
        axes2[i, 1].axis('off')

        im = axes2[i, 2].imshow(res['depth_map'], cmap='viridis')
        axes2[i, 2].set_title('Depth Map Dt\n(Depth Anything 3)', fontsize=10)
        axes2[i, 2].axis('off')
        plt.colorbar(im, ax=axes2[i, 2], fraction=0.046)

    plt.tight_layout()
    fig2.savefig(output_dir / 'fig4_2_road_mask_generation.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig4_2_road_mask_generation.png")

    # ====== FIGURE 3: 4.3.1 Point-level Anomaly Detection ======
    fig3, axes3 = plt.subplots(min(3, n_results), 3, figsize=(15, 5*min(3, n_results)))
    if min(3, n_results) == 1:
        axes3 = axes3.reshape(1, -1)
    fig3.suptitle('4.3.1 Point-level Anomaly Detection\n(Signed Height Difference)', fontsize=14, fontweight='bold')

    for i, res in enumerate(results[:min(3, n_results)]):
        frame = res['frame']
        road_mask = res['road_mask']
        signed_height = res['signed_height']

        axes3[i, 0].imshow(frame)
        overlay = road_mask > 0
        frame_with_road = frame.copy()
        frame_with_road[overlay] = (frame_with_road[overlay] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)
        axes3[i, 0].imshow(frame_with_road)
        axes3[i, 0].set_title(f'Frame {res["frame_idx"]}: Road Analysis', fontsize=10)
        axes3[i, 0].axis('off')

        im1 = axes3[i, 1].imshow(signed_height, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
        axes3[i, 1].set_title('Signed Height dh\n(dh = n*P + d)', fontsize=10)
        axes3[i, 1].axis('off')
        plt.colorbar(im1, ax=axes3[i, 1], fraction=0.046)

        axes3[i, 2].imshow(frame)
        anomaly_overlay = np.zeros_like(frame)
        anomaly_mask = res['anomaly_mask']
        anomaly_overlay[anomaly_mask > 0] = [255, 0, 0]
        axes3[i, 2].imshow(anomaly_overlay, alpha=0.7)
        axes3[i, 2].set_title('Anomaly Points (dh < theta_neg)\nAt = {p | dh < 0}', fontsize=10)
        axes3[i, 2].axis('off')

    plt.tight_layout()
    fig3.savefig(output_dir / 'fig4_3_1_point_anomaly_detection.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig4_3_1_point_anomaly_detection.png")

    # ====== FIGURE 4: 4.3.2 Cluster-level Candidate Formation ======
    fig4, axes4 = plt.subplots(min(3, n_results), 3, figsize=(15, 5*min(3, n_results)))
    if min(3, n_results) == 1:
        axes4 = axes4.reshape(1, -1)
    fig4.suptitle('4.3.2 Cluster-level Candidate Formation\n(Connected Component Analysis)', fontsize=14, fontweight='bold')

    for i, res in enumerate(results[:min(3, n_results)]):
        frame = res['frame']
        clusters = res['clusters']

        display_frame = frame.copy()
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))

        for j, cluster in enumerate(clusters):
            min_x, min_y, bw, bh = cluster['bbox']
            color = colors[j % len(colors)][:3]
            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=2,
                                    edgecolor=color, facecolor=color, alpha=0.3)
            axes4[i, 0].add_patch(rect)
            cx, cy = cluster['center']
            axes4[i, 0].plot(cx, cy, 'x', color=color, markersize=10, linewidth=2)

        axes4[i, 0].imshow(display_frame)
        axes4[i, 0].set_title(f'Frame {res["frame_idx"]}: Candidates ({len(clusters)} clusters)', fontsize=10)
        axes4[i, 0].axis('off')

        axes4[i, 1].imshow(frame)
        for j, cluster in enumerate(clusters[:5]):
            min_x, min_y, bw, bh = cluster['bbox']
            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=2,
                                    edgecolor='red', facecolor='yellow', alpha=0.4)
            axes4[i, 1].add_patch(rect)
            axes4[i, 1].text(min_x, min_y-5, f'C{cluster["id"]}', fontsize=9, color='white',
                             bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

        axes4[i, 1].set_title('Top 5 Candidates Detail', fontsize=10)
        axes4[i, 1].axis('off')

        axes4[i, 2].axis('off')
        if clusters:
            table_data = []
            for cluster in clusters[:5]:
                table_data.append([
                    f'C{cluster["id"]}',
                    f'{cluster["area"]}',
                    f'{cluster["negative_ratio"]:.2f}',
                    f'{cluster["mean_height"]:.3f}'
                ])
            table = axes4[i, 2].table(
                cellText=table_data,
                colLabels=['ID', 'Area(px)', 'Neg Ratio', 'Mean H'],
                loc='center', cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            axes4[i, 2].set_title('Candidate Properties', fontsize=10)
        else:
            axes4[i, 2].text(0.5, 0.5, 'No candidates', ha='center', va='center', fontsize=12)
            axes4[i, 2].axis('off')

    plt.tight_layout()
    fig4.savefig(output_dir / 'fig4_3_2_cluster_candidate_formation.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig4_3_2_cluster_candidate_formation.png")

    # ====== FIGURE 5: 4.3.3 Candidate Classification ======
    fig5, axes5 = plt.subplots(min(3, n_results), 3, figsize=(15, 5*min(3, n_results)))
    if min(3, n_results) == 1:
        axes5 = axes5.reshape(1, -1)
    fig5.suptitle('4.3.3 Candidate Classification\n(Curb vs Pothole)', fontsize=14, fontweight='bold')

    for i, res in enumerate(results[:min(3, n_results)]):
        frame = res['frame']
        clusters = res['clusters']

        display_frame = frame.copy()
        for cluster in clusters:
            min_x, min_y, bw, bh = cluster['bbox']
            obstacle_type = cluster.get('obstacle_type', 'unknown')
            color = (1.0, 0.65, 0.0) if obstacle_type == 'curb' else (0.54, 0.17, 0.89)
            label = 'CURB' if obstacle_type == 'curb' else 'POTHOLE'
            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=3,
                                    edgecolor=color, facecolor=color, alpha=0.4)
            axes5[i, 0].add_patch(rect)
            axes5[i, 0].text(min_x, min_y-8, label, fontsize=10, color='white',
                             bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

        axes5[i, 0].imshow(display_frame)
        axes5[i, 0].set_title(f'Frame {res["frame_idx"]}: Type Classification', fontsize=10)
        axes5[i, 0].axis('off')

        curbs = [c for c in clusters if c.get('obstacle_type') == 'curb']
        potholes = [c for c in clusters if c.get('obstacle_type') == 'pothole']

        axes5[i, 1].bar(['Curb', 'Pothole'], [len(curbs), len(potholes)],
                        color=['orange', 'purple'], alpha=0.7)
        axes5[i, 1].set_ylabel('Count')
        axes5[i, 1].set_title('Obstacle Type Distribution', fontsize=10)
        for j, v in enumerate([len(curbs), len(potholes)]):
            axes5[i, 1].text(j, v + 0.1, str(v), ha='center', fontsize=11)

        axes5[i, 2].text(0.1, 0.9, 'Classification Rules:', fontsize=11, fontweight='bold',
                         transform=axes5[i, 2].transAxes)
        rules = """
        Curb Rules:
        - Boundary ratio > 30%
        - Elongated (aspect > 2.5)

        Pothole Rules:
        - Negative ratio > 50%
        - Area > 100 px
        - Planar distribution
        """
        axes5[i, 2].text(0.1, 0.7, rules, fontsize=9, transform=axes5[i, 2].transAxes,
                         verticalalignment='top', family='monospace')
        axes5[i, 2].axis('off')

    plt.tight_layout()
    fig5.savefig(output_dir / 'fig4_3_3_candidate_classification.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig4_3_3_candidate_classification.png")

    # ====== FIGURE 6: Complete Pipeline ======
    fig6, axes6 = plt.subplots(2, 4, figsize=(20, 10))
    fig6.suptitle('Chapter 4 Complete Pipeline\n(Road Obstacle Detection)', fontsize=16, fontweight='bold')

    if results:
        res = results[0]
        frame = res['frame']
        h, w = frame.shape[:2]

        axes6[0, 0].imshow(frame)
        axes6[0, 0].set_title('(1) Input Image It', fontsize=11)
        axes6[0, 0].axis('off')

        road_display = np.zeros_like(frame)
        road_mask = res['road_mask']
        road_display[road_mask > 0] = [0, 200, 0]
        road_display[road_mask == 0] = frame[road_mask == 0] * 0.6
        axes6[0, 1].imshow(road_display)
        axes6[0, 1].set_title('(2) Road Mask Mt\n(PIDNet-S)', fontsize=11)
        axes6[0, 1].axis('off')

        im = axes6[0, 2].imshow(res['depth_map'], cmap='plasma')
        axes6[0, 2].set_title('(3) Depth Map Dt\n(Depth Anything 3)', fontsize=11)
        axes6[0, 2].axis('off')
        plt.colorbar(im, ax=axes6[0, 2], fraction=0.046)

        axes6[0, 3].imshow(frame)
        anomaly_overlay = np.zeros_like(frame)
        anomaly_overlay[res['anomaly_mask'] > 0] = [255, 0, 0]
        axes6[0, 3].imshow(anomaly_overlay, alpha=0.7)
        axes6[0, 3].set_title('(4) Anomaly Points At\n(dh < theta_neg)', fontsize=11)
        axes6[0, 3].axis('off')

        axes6[1, 0].imshow(frame)
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(res['clusters']), 1)))
        for j, cluster in enumerate(res['clusters']):
            min_x, min_y, bw, bh = cluster['bbox']
            color = colors[j % len(colors)][:3]
            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=2,
                                    edgecolor=color, facecolor=color, alpha=0.4)
            axes6[1, 0].add_patch(rect)
        axes6[1, 0].set_title(f'(5) Candidates Ct\n({len(res["clusters"])} clusters)', fontsize=11)
        axes6[1, 0].axis('off')

        axes6[1, 1].imshow(frame)
        for cluster in res['clusters']:
            min_x, min_y, bw, bh = cluster['bbox']
            obstacle_type = cluster.get('obstacle_type', 'unknown')
            color = (1.0, 0.65, 0.0) if obstacle_type == 'curb' else (0.54, 0.17, 0.89)
            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=2,
                                    edgecolor=color, facecolor=color, alpha=0.4)
            axes6[1, 1].add_patch(rect)
        axes6[1, 1].set_title('(6) Type Classification\n(Curb/Pothole)', fontsize=11)
        axes6[1, 1].axis('off')

        axes6[1, 2].imshow(frame)
        for cluster in res['clusters']:
            min_x, min_y, bw, bh = cluster['bbox']
            obstacle_type = cluster.get('obstacle_type', 'unknown')
            color = (1.0, 0.65, 0.0) if obstacle_type == 'curb' else (0.54, 0.17, 0.89)
            rect = patches.Rectangle((min_x, min_y), bw, bh, linewidth=2,
                                    edgecolor=color, facecolor='none')
            axes6[1, 2].add_patch(rect)
            axes6[1, 2].arrow(min_x + bw//2, min_y + bh//2, 30, 0,
                              head_width=5, head_length=5, fc='green', ec='green')
        axes6[1, 2].set_title('(7) Geometry Measurement\n(Dist/Width/Depth)', fontsize=11)
        axes6[1, 2].axis('off')

        axes6[1, 3].text(0.5, 0.8, 'Risk Assessment', fontsize=14, ha='center', fontweight='bold',
                         transform=axes6[1, 3].transAxes)
        axes6[1, 3].text(0.5, 0.6, f'Curb: {len(curbs)}', fontsize=11, ha='center',
                         transform=axes6[1, 3].transAxes, color='darkorange')
        axes6[1, 3].text(0.5, 0.45, f'Pothole: {len(potholes)}', fontsize=11, ha='center',
                         transform=axes6[1, 3].transAxes, color='indigo')
        axes6[1, 3].text(0.5, 0.25, '-> Output to Ch.5', fontsize=12, ha='center',
                         transform=axes6[1, 3].transAxes, color='green')
        axes6[1, 3].axis('off')

    plt.tight_layout()
    fig6.savefig(output_dir / 'fig4_complete_pipeline.png', dpi=150, bbox_inches='tight')
    print(f"Saved: fig4_complete_pipeline.png")

    # Generate summary report
    generate_summary_report(results, output_dir)
    print(f"\nAll results saved to: {output_dir}")
    return results

def generate_summary_report(results, output_dir):
    """Generate Chapter 4 summary report"""
    report = []
    report.append("# Chapter 4 Road Obstacle Detection - Experiment Summary")
    report.append("=" * 60)
    report.append("")
    report.append("## 4.1 Obstacle Type Definition")
    report.append("- Positive obstacle: Above ground plane, collision risk")
    report.append("- Negative obstacle: Below ground plane, fall/trap risk")
    report.append("- Focus: Negative obstacles (curb drop, pothole, depression)")
    report.append("")
    report.append("## 4.2 Image Preprocessing")
    report.append("- PIDNet-S outputs road mask Mt")
    report.append("- Limits analysis to road surface area")
    report.append("- Filters non-road regions (sky, buildings)")
    report.append("")
    report.append("## 4.3.1 Point-level Anomaly Detection")
    report.append("- Compute signed height: dh = n*P + d")
    report.append("- Mark point as anomaly when dh < theta_neg (threshold)")
    report.append("")
    report.append("## 4.3.2 Cluster-level Candidate Formation")
    report.append("- Use connected component analysis to group anomaly points")
    report.append("- Keep clusters with negative ratio > 20%")
    report.append("")
    report.append("## 4.3.3 Candidate Classification")
    report.append("- Curb: High boundary ratio + elongated shape")
    report.append("- Pothole: High negative ratio + planar distribution")
    report.append("")
    report.append("## Experiment Results")
    for res in results:
        report.append(f"\nFrame {res['frame_idx']}:")
        report.append(f"  - Anomaly points: {np.sum(res['anomaly_mask'] > 0)}")
        report.append(f"  - Candidate clusters: {len(res['clusters'])}")
        curbs = [c for c in res['clusters'] if c.get('obstacle_type') == 'curb']
        potholes = [c for c in res['clusters'] if c.get('obstacle_type') == 'pothole']
        report.append(f"  - Curb candidates: {len(curbs)}")
        report.append(f"  - Pothole candidates: {len(potholes)}")

    report_path = output_dir / 'chapter4_summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"Saved: chapter4_summary_report.txt")

def main():
    print("=" * 60)
    print("Chapter 4 Road Obstacle Detection - Visualization Demo")
    print("=" * 60)

    print("\n[1/3] Loading video frames...")
    frames = load_frames()
    print(f"Loaded {len(frames)} frames")

    if len(frames) == 0:
        print("Error: Cannot load frames")
        return

    print("\n[2/3] Generating visualizations...")
    results = visualize_chapter4(frames, OUTPUT_DIR)

    print("\n[3/3] Done!")
    return results

if __name__ == '__main__':
    main()
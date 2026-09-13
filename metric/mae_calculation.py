# Calculate for Mean Angular Error (MAE) between ground truth and predicted quaternions
import math

def calculate_mae(gt_list, pred_list):
    N = len(gt_list)
    total_error = 0
    
    for i in range(N):
        q1 = gt_list[i]
        q2 = pred_list[i]
        
        dot_product = (q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
        
        abs_dot = abs(dot_product)
        
        abs_dot = min(1.0, max(-1.0, abs_dot))
        
        theta_rad = 2 * math.acos(abs_dot)
        
        theta_deg = theta_rad * 180 / math.pi
        
        total_error += theta_deg

    return total_error / N

# --- Test Case ---
GT = [[1,0,0,0],[1,0,0,0],[1,0,0,0]]
MY = []

print("MAE:", calculate_mae(GT, MY))
import math

def append_sample_data(samples_array, cursor_x, cursor_y, predicted_x, predicted_y, predict_time, calibrator_time):
  samples_array.append({
    'cursor_x': cursor_x,
    'cursor_y': cursor_y,
    'predicted_x': predicted_x,
    'predicted_y': predicted_y,
    'predict_time': predict_time,
    'calibrator_time': calibrator_time
  })

def calculate_and_print_stats(samples_array):
  if not samples_array:
    print("No samples collected")
    return

  distances = []
  predict_times = []
  calibrator_times = []
  total_times = []

  for sample in samples_array:
    distance = math.hypot(
      sample['predicted_x'] - sample['cursor_x'],
      sample['predicted_y'] - sample['cursor_y']
    )
    distances.append(distance)
    predict_times.append(sample['predict_time'])
    calibrator_times.append(sample['calibrator_time'])
    total_times.append(sample['predict_time'] + sample['calibrator_time'])

  # Distance
  dist_min = min(distances)
  dist_max = max(distances)
  dist_avg = sum(distances) / len(distances)
  dist_std = (sum((d - dist_avg) ** 2 for d in distances) / len(distances)) ** 0.5

  # Predict time
  predict_min = min(predict_times)
  predict_max = max(predict_times)
  predict_avg = sum(predict_times) / len(predict_times)
  predict_std = (sum((t - predict_avg) ** 2 for t in predict_times) / len(predict_times)) ** 0.5
  predict_sum = sum(predict_times)

  # Calibrator time
  calibrator_min = min(calibrator_times)
  calibrator_max = max(calibrator_times)
  calibrator_avg = sum(calibrator_times) / len(calibrator_times)
  calibrator_std = (sum((t - calibrator_avg) ** 2 for t in calibrator_times) / len(calibrator_times)) ** 0.5
  calibrator_sum = sum(calibrator_times)

  # Total time
  total_min = min(total_times)
  total_max = max(total_times)
  total_avg = sum(total_times) / len(total_times)
  total_std = (sum((t - total_avg) ** 2 for t in total_times) / len(total_times)) ** 0.5
  total_sum = sum(total_times)

  print(f"\n=== Statistics for {len(samples_array)} samples ===")
  print(f"Distance: min={dist_min:.2f}, max={dist_max:.2f}, avg={dist_avg:.2f}, std={dist_std:.2f}")
  print(
    f"Predict time: sum={predict_sum:.2f}ms, min={predict_min:.2f}ms, max={predict_max:.2f}ms, avg={predict_avg:.2f}ms, std={predict_std:.2f}ms")
  print(
    f"Calibrator time: sum={calibrator_sum:.2f}ms, min={calibrator_min:.2f}ms, max={calibrator_max:.2f}ms, avg={calibrator_avg:.2f}ms, std={calibrator_std:.2f}ms")
  print(
    f"Total time: sum={total_sum:.2f}ms, min={total_min:.2f}ms, max={total_max:.2f}ms, avg={total_avg:.2f}ms, std={total_std:.2f}ms")

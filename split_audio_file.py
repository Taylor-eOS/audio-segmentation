import librosa
import numpy as np
import soundfile as sf
import os
from scipy.signal import medfilt

def otsu_threshold(data):
    """Compute Otsu's threshold for 1D data"""
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    max_var = 0
    best_thresh = 0
    
    for i in range(1, len(bin_edges)):
        w0 = hist[:i].sum() / total
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (hist[:i] * bin_centers[:i]).sum() / w0
        mu1 = (hist[i:] * bin_centers[i:]).sum() / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > max_var:
            max_var = var
            best_thresh = bin_edges[i]
    return best_thresh

def compute_speech_features(y, sr):
    """Compute RMS energy and spectral spread with adaptive framing"""
    frame_length = int(sr * 0.05)  # 50ms frames
    hop_length = int(frame_length / 2)  # 25ms hop
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_spread = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                        n_fft=frame_length, hop_length=hop_length)[0]
    
    # Smooth features with median filter
    rms = medfilt(rms, kernel_size=5)
    spectral_spread = medfilt(spectral_spread, kernel_size=5)
    
    return rms, spectral_spread, frame_length, hop_length

def find_initial_splits(y, sr, rms, frame_length, hop_length):
    """Initial split point detection with adaptive thresholding"""
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, 
                                  hop_length=hop_length, n_fft=frame_length)
    
    # Dynamic RMS threshold using Otsu
    rms_threshold = otsu_threshold(rms)
    silent_regions = rms < rms_threshold
    
    # Detect pause durations
    pauses = []
    current_pause = []
    for i, is_silent in enumerate(silent_regions):
        if is_silent:
            current_pause.append(i)
        else:
            if current_pause:
                pauses.append((current_pause[0], current_pause[-1]))
                current_pause = []
    
    pause_durations = [(times[end] - times[start]) for start, end in pauses]
    if pause_durations:
        pause_threshold = max(0.2, otsu_threshold(np.array(pause_durations)) * 0.8)
    else:
        pause_threshold = 0.3
    
    # Find valid split points
    split_points = [0.0]
    for start, end in pauses:
        duration = times[end] - times[start]
        if duration >= pause_threshold:
            split_time = (times[start] + times[end]) / 2
            if split_time - split_points[-1] > 5:  # Minimum 5s between splits
                split_points.append(split_time)
    
    split_points.append(librosa.get_duration(y=y, sr=sr))
    return sorted(list(set(split_points)))

def refine_split_points(y, sr, initial_splits, max_iterations=3):
    """Iteratively refine split points focusing on long segments"""
    splits = initial_splits.copy()
    for _ in range(max_iterations):
        new_splits = []
        for i in range(len(splits)-1):
            start, end = splits[i], splits[i+1]
            duration = end - start
            
            if duration <= 30:
                new_splits.extend([start, end])
                continue
                
            # Process long segment with more sensitive parameters
            segment = y[int(start*sr):int(end*sr)]
            rms, _, fl, hl = compute_speech_features(segment, sr)
            
            if len(rms) < 10:  # Too short for meaningful analysis
                new_splits.extend([start, end])
                continue
                
            # Adaptive threshold for sub-segment
            local_thresh = otsu_threshold(rms) * 0.7
            sub_silent = rms < local_thresh
            
            # Find potential splits in sub-segment
            sub_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, 
                                              hop_length=hl, n_fft=fl)
            candidate_splits = []
            current = []
            for j, silent in enumerate(sub_silent):
                if silent:
                    current.append(j)
                else:
                    if current:
                        dur = sub_times[current[-1]] - sub_times[current[0]]
                        if dur >= 0.2:
                            candidate_splits.append(sub_times[current[0]] + start)
                    current = []
            
            # Add best candidate near target duration (25s)
            target = start + 25
            best_split = None
            min_diff = float('inf')
            for s in candidate_splits:
                if start < s < end:
                    diff = abs((s - start) - 25)
                    if diff < min_diff:
                        best_split = s
                        min_diff = diff
            
            if best_split and (best_split - start > 10) and (end - best_split > 10):
                new_splits.extend([start, best_split, end])
            else:
                new_splits.extend([start, end])
        
        # Update splits and remove duplicates
        splits = sorted(list(set(new_splits)))
    
    return splits

def merge_short_segments(splits, min_duration=20, max_duration=30):
    """Merge segments shorter than minimum duration"""
    final_splits = [splits[0]]
    for split in splits[1:]:
        prev = final_splits[-1]
        current_duration = split - prev
        
        if current_duration < min_duration:
            # Check if merging with next would stay under max duration
            if len(final_splits) > 1:
                prev_prev = final_splits[-2]
                merged_duration = split - prev_prev
                if merged_duration <= max_duration:
                    final_splits.pop()
            else:
                final_splits.append(split)
        else:
            final_splits.append(split)
    
    return final_splits

def split_audio_file(input_path, output_dir, sr=22050):
    """Main processing function"""
    # Load and preprocess audio
    y, sr = librosa.load(input_path, sr=sr, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Initial feature extraction
    rms, spectral_spread, fl, hl = compute_speech_features(y, sr)
    
    # Initial split detection
    initial_splits = find_initial_splits(y, sr, rms, fl, hl)
    
    # Iterative refinement
    refined_splits = refine_split_points(y, sr, initial_splits)
    
    # Final merging of short segments
    final_splits = merge_short_segments(refined_splits)
    
    # Ensure final split point matches audio duration
    if final_splits[-1] < total_duration - 0.1:  # Account for float precision
        final_splits.append(total_duration)
    final_splits[-1] = total_duration  # Ensure exact match
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split and export segments
    for i, (start, end) in enumerate(zip(final_splits[:-1], final_splits[1:])):
        duration = end - start
        if duration < 1:  # Skip very short remaining segments
            continue
        
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        
        output_path = os.path.join(output_dir, f"segment_{i+1:03d}_{duration:.1f}s.wav")
        sf.write(output_path, segment, sr)
    
    return final_splits

# Example usage
if __name__ == "__main__":
    input_file = "input_audio.wav"
    output_directory = "output_segments"
    split_points = split_audio_file(input_file, output_directory)
    print(f"Created {len(split_points)-1} segments at:\n{split_points}")


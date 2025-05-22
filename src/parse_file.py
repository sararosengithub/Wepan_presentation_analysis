import csv
from pprint import pprint
from scipy.spatial.distance import euclidean
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

def parse_event_file(file_path):
    """
    Parses an event file with dynamic headers.

    Args:
        file_path (str): The path to the .event file.

    Returns:
        tuple: (header_definitions, events_data)
               header_definitions (dict): A dictionary mapping (category, event_type)
                                          tuples to a list of their field names.
               events_data (list): A list of dictionaries, where each dictionary
                                   represents an event with its parsed fields.
    """
    header_definitions = {}
    events_data = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row_parts in reader:
            if not row_parts:  
                continue

            if row_parts[0] == "HEADER":
                category = row_parts[1]
                event_sub_type = row_parts[2]
                field_names = row_parts[3:]
                header_definitions[(category, event_sub_type)] = field_names
            else:
                timestamp = int(row_parts[0]) 
                category = row_parts[1]
                event_sub_type = row_parts[2]
                values = row_parts[3:]

                event_record = {
                    "timestamp": timestamp,
                    "category": category,
                    "event_type": event_sub_type  
                }

                field_names = header_definitions.get((category, event_sub_type))

                if field_names:
                    if len(values) == len(field_names):
                        for i, field_name in enumerate(field_names):
                            event_record[field_name] = values[i]
                    else:
                        print(f"Warning: Mismatch in field count for event {category}/{event_sub_type} at timestamp {timestamp}.")
                        print(f"  Expected {len(field_names)} fields ({field_names}), got {len(values)} ({values}).")
                        for i, field_name in enumerate(field_names):
                            if i < len(values):
                                event_record[field_name] = values[i]
                            else:
                                event_record[field_name] = None
                else:
                    print(f"Warning: No header definition found for event {category}/{event_sub_type} at timestamp {timestamp}.")
                    event_record['raw_values'] = values

                events_data.append(event_record)

    return header_definitions, events_data


def load_and_prepare_scenario_data(scenario_id, event_file_path, frame_file_path):
    print(f"\n--- Loading and Preparing Scenario: {scenario_id} ---")
    print(f"Event File: {event_file_path}")
    print(f"Frame File: {frame_file_path}")

    events_list = []
    current_frame_df = pd.DataFrame()
    current_drills_for_analysis = []

    try:
        _, events_list = parse_event_file(event_file_path)
        current_frame_df = pd.read_csv(frame_file_path)
        current_frame_df['timestamp'] = pd.to_numeric(current_frame_df['timestamp'], errors='coerce')
        current_frame_df.dropna(subset=['timestamp'], inplace=True)
        current_frame_df['timestamp'] = current_frame_df['timestamp'].astype(np.int64)
    except FileNotFoundError as e:
        print(f"Error: File not found for scenario {scenario_id} - {e}. Skipping.")
        return None, pd.DataFrame(), [] # Return empty structures
    except Exception as e:
        print(f"Error loading data for scenario {scenario_id} - {e}. Skipping.")
        return None, pd.DataFrame(), []

    # Prepare drills for analysis (specific to this scenario's events_list)
    drills_data_map = {}
    for event in events_list:
        event_timestamp = int(event['timestamp'])
        if event['event_type'] == 'drill_start':
            drill_uid = event['drill_uid']
            drills_data_map.setdefault(drill_uid, {}).update({
                'drill_uid': drill_uid, 'start_time': event_timestamp, 'shots_raw': []
            })
        elif event['event_type'] == 'drill_end':
            drill_uid = event['drill_uid']
            if drill_uid in drills_data_map: drills_data_map[drill_uid]['end_time'] = event_timestamp
    
    all_shot_events = [event for event in events_list if event['event_type'] == 'drill_shot']
    all_shot_events.sort(key=lambda x: int(x['timestamp']))
    for shot_event in all_shot_events:
        shot_time, shot_idx = int(shot_event['timestamp']), shot_event.get('shot_index')
        for drill_uid, drill_info in drills_data_map.items(): # Iterate over a copy of items for safety if modifying
            if 'start_time' not in drill_info: continue
            if shot_time >= drill_info['start_time'] and \
               (drill_info.get('end_time') is None or shot_time <= drill_info['end_time']):
                drill_info['shots_raw'].append({'timestamp': shot_time, 'shot_index': shot_idx}); break
    
    for drill_uid, drill_info in drills_data_map.items():
        drill_info.get('shots_raw', []).sort(key=lambda s: (s['timestamp'], s.get('shot_index', '')))
        first_shot_time = next((s['timestamp'] for s in drill_info.get('shots_raw', []) if s.get('shot_index') == '0'), None)
        if 'start_time' in drill_info and first_shot_time is not None:
            current_drills_for_analysis.append({
                'drill_uid': drill_uid, 'start_time': drill_info['start_time'],
                'end_time': drill_info.get('end_time'), 'first_shot_time': first_shot_time
            })
    current_drills_for_analysis.sort(key=lambda x: x['start_time'])
    
    return events_list, current_frame_df, current_drills_for_analysis
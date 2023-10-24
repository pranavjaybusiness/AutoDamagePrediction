import csv

class ReportProcessor:

    @staticmethod
    def print_feature_count(input_file, output_file, feature_name):
        feature_counts = {}

        # Read the input file
        with open(input_file, 'r') as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                value = row.get(feature_name, '').strip()
                if value:
                    feature_counts[value] = feature_counts.get(value, 0) + 1
                    print(f"Identified {value}")  # Debugging print statement

        # Check if feature_counts is empty
        if not feature_counts:
            print(f"No {feature_name} found in the input file.")
            return

        # Write to the output file
        with open(output_file, 'w') as file:
            for key, count in feature_counts.items():
                file.write(f"{key}:{count}\n")

    def print_collision_type(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Collision Type')

    def print_weather(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Weather')

    def print_driver_at_fault(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Driver At Fault')

    def print_vehicle_first_impact_location(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Vehicle First Impact Location')

    def print_vehicle_second_impact_location(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Vehicle Second Impact Location')

    def print_vehicle_body_type(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Vehicle Body Type')

    def print_vehicle_movement(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Vehicle Movement')

    def print_speed_limit(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Speed Limit')
    def print_veichle_damage_extent(self, input_file, output_file):
        self.print_feature_count(input_file, output_file, 'Vehicle Damage Extent')
    def print_specific_counts(self, input_file):
        count = 0
        with open(input_file, 'r') as file:
            reader = csv.DictReader(file, delimiter=',')
            for row in reader:
                car_model = row.get('Vehicle Model', '').strip()
                speed_limit = row.get('Speed Limit', '').strip()
                severity = row.get('Vehicle Damage Extent','').strip()
                movement = row.get('Vehicle Movement','').strip()

                if car_model == "TK" and speed_limit == "65" and movement == "MOVING CONSTANT SPEED" and severity == "OTHER":
                    count += 1
        print(f"Car Model TK occurred {count} times in the file with a Speed Limit of 45.")

# Usage
processor = ReportProcessor()
#processor.print_veichle_damage_extent('Data/Data.txt', 'Parsed/damage_extent_For_Data.txt')
processor.print_specific_counts('Data/NewYorkData.txt')

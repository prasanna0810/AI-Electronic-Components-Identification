import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Define class labels (matching dataset naming format)
CLASS_NAMES = [
    "Capacitors_aluminum_electrolytic", "analog_multipliers__dividers", "automotive_relays",
    "batteries_non_rechargeable__primary_", "batteries_rechargeable__secondary_", "circuit_breakers",
    "fiber_optic_cables", "fixed_inductors", "fuseholders", "fuses",
    "instrumentation__op_amps__buffer_amps", "isolation_transformers_and_autotransformers__step_up__step_down",
    "led_character_and_numeric", "mica_and_ptfe_capacitors", "motors___ac__dc",
    "multimeters", "pliers", "power_supplies__test__bench_", "pushbutton_switches",
    "rocker_switches", "rotary_potentiometers__rheostats", "rotary_switches",
    "single_bipolar_transistors", "single_diodes", "solar_cells",
    "stepper_motors", "strain_gauges", "through_hole_resistors",
    "toggle_switches", "tweezers", "ultrasonic_receivers__transmitters",
    "usb_cables", "video_cables__dvi__hdmi_", "wrenches"
]

# Detailed descriptions
COMPONENT_DETAILS = {
    "Capacitors_aluminum_electrolytic": "Used for energy storage and signal filtering in power supplies and audio circuits.",
    "analog_multipliers__dividers": "Performs mathematical functions like signal scaling and modulation in analog systems.",
    "automotive_relays": "Electromechanical switches used to control high-power automotive components.",
    "batteries_non_rechargeable__primary_": "Single-use batteries for remote controls, medical devices, and emergency electronics.",
    "batteries_rechargeable__secondary_": "Rechargeable power sources for electronics, laptops, and electric vehicles.",
    "circuit_breakers": "Protects circuits by automatically disconnecting power during overloads.",
    "fiber_optic_cables": "Transmits high-speed data using light pulses, immune to electromagnetic interference.",
    "fixed_inductors": "Stores energy and filters signals in electrical circuits.",
    "fuseholders": "Provides a secure mount for fuses in electrical circuits.",
    "fuses": "Prevents circuit damage by breaking when excess current flows.",
    "instrumentation__op_amps__buffer_amps": "Amplifies signals in measurement, control, and audio applications.",
    "isolation_transformers_and_autotransformers__step_up__step_down": "Converts voltage levels and provides electrical isolation.",
    "led_character_and_numeric": "Displays alphanumeric information in digital electronics.",
    "mica_and_ptfe_capacitors": "High-frequency capacitors with excellent stability and low loss.",
    "motors___ac__dc": "Convert electrical energy into mechanical motion for automation and robotics.",
    "multimeters": "Measures voltage, current, and resistance in electrical circuits.",
    "pliers": "Hand tools used for gripping and cutting wires in electronics work.",
    "power_supplies__test__bench_": "Provides controlled DC voltage for testing electronic circuits.",
    "pushbutton_switches": "Momentary or latching switches used in control panels and devices.",
    "rocker_switches": "Common on/off switches used in home appliances and electrical devices.",
    "rotary_potentiometers__rheostats": "Adjustable resistors for tuning voltage or current in circuits.",
    "rotary_switches": "Mechanical switches for selecting different circuit paths.",
    "single_bipolar_transistors": "Amplifies and switches electrical signals in circuits.",
    "single_diodes": "Allows current to flow in one direction, used in rectifiers and signal processing.",
    "solar_cells": "Converts sunlight into electrical energy for renewable power applications.",
    "stepper_motors": "Precision motors used in CNC machines, robotics, and automation.",
    "strain_gauges": "Measures strain and mechanical stress in structures.",
    "through_hole_resistors": "Fixed-value resistors used for controlling current in circuits.",
    "toggle_switches": "Manually operated switches for turning circuits on and off.",
    "tweezers": "Used for handling small components in precision electronics work.",
    "ultrasonic_receivers__transmitters": "Convert sound waves to electrical signals, used in sensors and robotics.",
    "usb_cables": "Used for data transfer and power delivery between electronic devices.",
    "video_cables__dvi__hdmi_": "Transmits high-definition video signals between devices.",
    "wrenches": "Mechanical tool for assembling and disassembling electronic devices."
}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_component(image_path):
    """
    Loads an image, classifies the electronic component, and provides details.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform classification
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = CLASS_NAMES[predicted.item()]
    
    # Retrieve details
    description = COMPONENT_DETAILS.get(class_name, "No additional details available.")
    
    # Display results
    print(f"🛠 **Predicted Component**: {class_name}")
    print(f"ℹ **Description**: {description}")

# Example Usage
if __name__ == "__main__":
    image_path = r"C:\Users\prasa\Downloads\hc-sr04_ultrasonic_sensor_distance_measuring_module_1_.jpg"  # Update with actual test image path
    classify_component(image_path)

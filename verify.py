import neuronguard as ng
import time

print("🧠 Initializing Local Biological Cortex (1000 Sensory -> 4 Motor)...")
# Footprint allocates ~62.75 KB, fitting entirely within native CPU caches
cortex = ng.NeuronGuardField(sensory_count=1000, motor_count=4)

# Simulated streaming ingestion loop (representing terabytes of data rows)
print("🚀 Simulating high-velocity streaming reflex ticks...")
for step in range(5):
    # Simulating a sudden burst of sensory stimuli tokens arriving over the wire
    active_stimuli = [42, 108, 512]

    # Process stream drops the GIL instantly to run raw Rust address logic
    triggered_motor_id = cortex.process_stream(active_stimuli, training_mode=True)
    print(f"  [Tick {step}] Stimuli {active_stimuli} ➔ Triggered Motor Neuron Terminal: {triggered_motor_id}")

    # Fire your prototype's metabolic decay loop to shave off old electrical energy
    cortex.tick_decay(0.90)
    time.sleep(0.1)

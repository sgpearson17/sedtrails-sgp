from logger import log_simulation_state, log_exception

def run_simulation():
    try:
        state = {"step": 1, "value": 100, "status": "running"}
        log_simulation_state(state)

        # Simulate a failure
        state["step"] = 2
        state["value"] = 50
        log_simulation_state(state)

        raise ValueError("Unexpected simulation failure!")  # Intentional error

    except Exception as e:
        log_exception(e)

# Run the simulation
if __name__ == "__main__":
    run_simulation()

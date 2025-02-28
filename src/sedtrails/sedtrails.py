import time
import yaml
import typer


class SedtrailsCLI:
    def __init__(self):
        # Create the Typer app instance
        self.app = typer.Typer()
        # Register CLI commands
        self.register_commands()

    def load_config(self, config_path: str) -> dict:
        """
        Loads configuration from a YAML file.
        """
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            typer.echo(f"Error loading config from {config_path}: {e}")
            raise typer.Exit(1)

    def register_commands(self):
        @self.app.command()
        def run_loop(config: str = "config.yml"):
            """
            Runs a time loop based on configuration from a YAML file.
            The YAML file should include 'interval' (seconds) and 'iterations'.
            """
            conf = self.load_config(config)
            # default to 1 second if not specified
            interval = conf.get("interval", 1)
            # default to 10 iterations if not specified
            iterations = conf.get("iterations", 10)
            typer.echo(f"Starting time loop with {iterations} iterations and \
                       {interval} seconds interval.")
            for i in range(iterations):
                typer.echo(f"Iteration {i + 1}")
                time.sleep(interval)
            typer.echo("Time loop completed.")

        @self.app.command()
        def close():
            """
            Closes the application gracefully.
            """
            typer.echo("Closing the application. Goodbye!")
            raise typer.Exit()

    def run(self):
        # Run the Typer application, which parses command line arguments and
        # executes commands
        self.app()


if __name__ == "__main__":
    cli = SedtrailsCLI()
    cli.run()

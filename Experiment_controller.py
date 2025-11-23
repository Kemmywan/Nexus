#!/usr/bin/env python3
"""
Experiment Controller Script
Control experiment steps and log results to markdown files
"""

import os
import sys
import subprocess
import datetime
import argparse
import json
from pathlib import Path

class ExperimentController:
    def __init__(self):
        self.current_experiment = None
        self.output_file = None
        self.config_file = "experiment_config.json"
        self.load_config()
        
    def load_config(self):
        """Load or create experiment configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.current_experiment = config.get('current_experiment')
                self.output_file = config.get('output_file')
        else:
            self.save_config()
    
    def save_config(self):
        """Save current experiment configuration"""
        config = {
            'current_experiment': self.current_experiment,
            'output_file': self.output_file
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def start_experiment(self, experiment_name=None):
        """Start a new experiment and create output markdown file"""
        if not experiment_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.current_experiment = experiment_name
        
        # Create experiments directory if needed
        os.makedirs("experiments", exist_ok=True)
        self.output_file = f"experiments/{experiment_name}.md"
        
        # Initialize markdown file
        with open(self.output_file, 'w') as f:
            f.write(f"# Experiment: {experiment_name}\n\n")
            f.write(f"**Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Working Directory:** {os.getcwd()}\n\n")
            f.write("## Experiment Configuration\n\n")
            f.write("```json\n")
            f.write("{\n")
            f.write(f'  "experiment_name": "{experiment_name}",\n')
            f.write(f'  "start_time": "{datetime.datetime.now().isoformat()}",\n')
            f.write(f'  "working_dir": "{os.getcwd()}"\n')
            f.write("}\n```\n\n")
            f.write("## Experiment Log\n\n")
        
        self.save_config()
        print(f"✅ Started experiment: {experiment_name}")
        print(f"📝 Output file: {self.output_file}")
        return self.output_file
    
    def run_command(self, command, description=None, timeout=None):
        """Run a bash command and log results to markdown"""
        if not self.current_experiment:
            print("❌ No active experiment. Start one first with 'start' command.")
            return False
        
        print(f"🚀 Running: {command}")
        
        # Log command to markdown
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        with open(self.output_file, 'a') as f:
            f.write(f"### {timestamp} - ")
            if description:
                f.write(f"{description}\n\n")
            else:
                f.write("Command Execution\n\n")
            
            f.write(f"**Command:** `{command}`\n")
            f.write(f"**Directory:** `{os.getcwd()}`\n")
            f.write(f"**Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        try:
            # Run command and capture output
            start_time = datetime.datetime.now()
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd(),
                timeout=timeout
            )
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log results
            with open(self.output_file, 'a') as f:
                f.write(f"**Duration:** {duration:.2f} seconds\n")
                f.write(f"**Exit Code:** {result.returncode}\n\n")
                
                if result.stdout:
                    f.write("**Output:**\n```\n")
                    f.write(result.stdout)
                    f.write("\n```\n\n")
                
                if result.stderr:
                    f.write("**Errors:**\n```\n")
                    f.write(result.stderr)
                    f.write("\n```\n\n")
                
                f.write("---\n\n")
            
            # Print summary
            if result.returncode == 0:
                print(f"✅ Command completed successfully ({duration:.2f}s)")
            else:
                print(f"❌ Command failed with exit code {result.returncode} ({duration:.2f}s)")
            
            if result.stdout:
                preview = result.stdout[:300] + "..." if len(result.stdout) > 300 else result.stdout
                print(f"📤 Output: {preview}")
            if result.stderr:
                preview = result.stderr[:300] + "..." if len(result.stderr) > 300 else result.stderr
                print(f"⚠️ Errors: {preview}")
                
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds"
            print(f"⏰ {error_msg}")
            
            with open(self.output_file, 'a') as f:
                f.write(f"**Status:** TIMEOUT ({timeout}s)\n\n")
                f.write("---\n\n")
            
            return False
            
        except Exception as e:
            error_msg = f"Exception occurred: {str(e)}"
            print(f"❌ {error_msg}")
            
            with open(self.output_file, 'a') as f:
                f.write(f"**Exception:** {error_msg}\n\n")
                f.write("---\n\n")
            
            return False
    
    def add_note(self, note):
        """Add a custom note to the experiment log"""
        if not self.current_experiment:
            print("❌ No active experiment. Start one first.")
            return
        
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        with open(self.output_file, 'a') as f:
            f.write(f"### {timestamp} - 📝 Note\n\n")
            f.write(f"{note}\n\n")
            f.write("---\n\n")
        
        print("📝 Note added to experiment log")
    
    def add_section(self, title, content=None):
        """Add a new section to the experiment log"""
        if not self.current_experiment:
            print("❌ No active experiment. Start one first.")
            return
        
        with open(self.output_file, 'a') as f:
            f.write(f"## {title}\n\n")
            if content:
                f.write(f"{content}\n\n")
            f.write("---\n\n")
        
        print(f"📋 Section '{title}' added to experiment log")
    
    def list_experiments(self):
        """List all experiment files"""
        exp_dir = Path("experiments")
        if not exp_dir.exists():
            print("No experiments directory found.")
            return
        
        md_files = list(exp_dir.glob("*.md"))
        if not md_files:
            print("No experiment files found.")
            return
        
        print("📁 Available experiments:")
        for f in sorted(md_files, key=lambda x: x.stat().st_mtime, reverse=True):
            size = f.stat().st_size
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  - {f.name} ({size} bytes, modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    def status(self):
        """Show current experiment status"""
        if self.current_experiment:
            print(f"🧪 Active experiment: {self.current_experiment}")
            print(f"📝 Output file: {self.output_file}")
            if os.path.exists(self.output_file):
                size = os.path.getsize(self.output_file)
                print(f"📊 File size: {size} bytes")
        else:
            print("❌ No active experiment")
    
    def end_experiment(self):
        """End current experiment"""
        if not self.current_experiment:
            print("❌ No active experiment to end.")
            return
        
        with open(self.output_file, 'a') as f:
            f.write(f"## Experiment Completed\n\n")
            f.write(f"**Ended:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        print(f"🏁 Experiment '{self.current_experiment}' completed")
        print(f"📝 Final report: {self.output_file}")
        
        self.current_experiment = None
        self.output_file = None
        self.save_config()
    
    def interactive_mode(self):
        """Interactive mode for running experiments"""
        print("🧪 Experiment Controller - Interactive Mode")
        print("Commands: start [name], run <command>, note <text>, section <title>, status, list, end, quit")
        print("Use 'help' for detailed command information")
        
        while True:
            try:
                prompt = f"[{self.current_experiment or 'No experiment'}] > "
                cmd = input(prompt).strip()
                
                if cmd.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif cmd.startswith('start'):
                    parts = cmd.split(maxsplit=1)
                    name = parts[1] if len(parts) > 1 else None
                    self.start_experiment(name)
                
                elif cmd.startswith('run '):
                    # Parse: run <command> [# description]
                    rest = cmd[4:].strip()
                    if ' # ' in rest:
                        command, description = rest.split(' # ', 1)
                    else:
                        command, description = rest, None
                    self.run_command(command.strip(), description.strip() if description else None)
                
                elif cmd.startswith('note '):
                    note = cmd[5:].strip()
                    self.add_note(note)
                
                elif cmd.startswith('section '):
                    title = cmd[8:].strip()
                    self.add_section(title)
                
                elif cmd == 'status':
                    self.status()
                
                elif cmd == 'list':
                    self.list_experiments()
                
                elif cmd == 'end':
                    self.end_experiment()
                
                elif cmd == 'help':
                    print("\n📖 Available Commands:")
                    print("  start [name]           - Start new experiment (auto-named if no name given)")
                    print("  run <cmd> [# desc]     - Run bash command with optional description")
                    print("  note <text>            - Add note to experiment log")
                    print("  section <title>        - Add new section to experiment")
                    print("  status                 - Show current experiment status")
                    print("  end                    - End current experiment")
                    print("  list                   - List all experiments")
                    print("  quit                   - Exit interactive mode")
                    print("\n💡 Tips:")
                    print("  - Use 'run python train.py # Training model' to add descriptions")
                    print("  - Experiment logs are saved in 'experiments/' directory")
                    print("  - Use 'section Results' to organize your experiment report\n")
                
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Experiment Controller - Control and log your ML experiments')
    parser.add_argument('--start', type=str, help='Start experiment with name')
    parser.add_argument('--run', type=str, help='Run command')
    parser.add_argument('--desc', type=str, help='Description for command')
    parser.add_argument('--note', type=str, help='Add note')
    parser.add_argument('--section', type=str, help='Add section')
    parser.add_argument('--list', action='store_true', help='List experiments')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--end', action='store_true', help='End current experiment')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    controller = ExperimentController()
    
    if args.interactive or len(sys.argv) == 1:
        controller.interactive_mode()
    elif args.start:
        controller.start_experiment(args.start)
    elif args.run:
        controller.run_command(args.run, args.desc)
    elif args.note:
        controller.add_note(args.note)
    elif args.section:
        controller.add_section(args.section)
    elif args.status:
        controller.status()
    elif args.end:
        controller.end_experiment()
    elif args.list:
        controller.list_experiments()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
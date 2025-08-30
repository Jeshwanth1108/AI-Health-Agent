from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam

# Load environment variables including your API key
load_dotenv(dotenv_path='apikey.env')
os.environ["OPENAI_API_KEY"] = os.getenv("APIKEY")

# Read the medical report file using forward slashes (Linux)
with open("Medical Reports/Medical Rerort - Michael Johnson - Panic Attack Disorder.txt", "r") as file:
    medical_report = file.read()

# Initialize agents
agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}

# Function to run each agent and get their response
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

# Run agents concurrently and collect their responses
responses = {}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    
    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response

# Initialize multidisciplinary team agent with gathered reports
team_agent = MultidisciplinaryTeam(
    cardiologist_report=responses.get("Cardiologist", ""),
    psychologist_report=responses.get("Psychologist", ""),
    pulmonologist_report=responses.get("Pulmonologist", "")
)

# Run team agent to generate the final diagnosis
final_diagnosis = team_agent.run()

# Prepare output directory path and ensure it exists
output_dir = "Results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "final_diagnosis.txt")

# Write the final diagnosis to the file
with open(output_path, "w") as txt_file:
    txt_file.write("### Final Diagnosis:\n\n")
    txt_file.write(final_diagnosis)

print(f"Final diagnosis has been saved to {output_path}")


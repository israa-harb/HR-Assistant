from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables

load_dotenv()  # load from .env

# OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")

# Optional: LangChain / LangSmith API key
langchain_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "HR assistant"
else:
    print("Warning: LANGCHAIN_API_KEY not found. LangSmith tracing will be disabled.")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# -------------------------
# Initialize LLM
# -------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key)

# -------------------------
# Define tools
# -------------------------
def get_employee_name(emp_id: str) -> dict:
    if emp_id == "123":
        return {"name": "Smith", "role": "Manager", "Department": "IT"}
    elif emp_id == "453":
        return {"name": "Paul", "role": "Supervisor", "Department": "CCE"}
    else:
        return {}

def check_leave_balance(emp_id: str) -> dict:
    if emp_id == "123":
        return {"remaining_days": "10", "used_days": "30"}
    elif emp_id == "453":
        return {"remaining_days": "2", "used_days": "38"}
    else:
        return {}

def generate_interview_question(role: str) -> list:
    if role == "Computer Engineering":
        return [
            "Can you explain the differences between a process and a thread?",
            "How does a CPU handle interrupts, and why are they important?",
            "Describe how virtual memory works and why it's used.",
            "What are the key differences between RISC and CISC architectures?",
            "How would you optimize code for performance at the hardware level?"
        ]
    elif role == "AI Engineering":
        return [
            "Explain the difference between supervised, unsupervised, and reinforcement learning.",
            "How do you handle imbalanced datasets in machine learning?",
            "What is the bias-variance tradeoff?",
            "Describe a project where you deployed a machine learning model in production.",
            "How would you select the right model for a classification problem?"
        ]
    elif role == "Biomedical Engineering":
        return [
            "What biomedical devices have you worked with, and what challenges did you face?",
            "Explain how you would validate the accuracy and safety of a medical sensor.",
            "Describe a time you collaborated with clinicians or patients during a project.",
            "What is your experience with regulatory compliance in biomedical design?",
            "How would you improve the design of a prosthetic limb for better user comfort?"
        ]
    else:
        return ["No questions available for this role."]

# -------------------------
# Wrap tools for agent
# -------------------------
emp_name_tool = Tool.from_function(
    name="check_employee_information",
    func=get_employee_name,
    description="Retrieve an employee's name, role, and department by providing their employee ID."
)

leave_balance_tool = Tool.from_function(
    name="check_leave_balance",
    func=check_leave_balance,
    description="Check how many leave days an employee has by providing the employee ID."
)

interview_tool = Tool.from_function(
    name="generate_interview_questions",
    func=generate_interview_question,
    description="Generate interview questions for a given job role."
)

# -------------------------
# Initialize agent
# -------------------------
agent = initialize_agent(
    tools=[emp_name_tool, leave_balance_tool, interview_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# -------------------------
# Streamlit app
# -------------------------
st.set_page_config(page_title="HR-Assistant", layout="centered")
st.title("Hello! I am your HR Assistant")

question = st.text_input("Enter your question")

if question:
    try:
        response = agent.run(question)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("Please enter a question.")


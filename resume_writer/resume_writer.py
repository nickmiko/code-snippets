import os
from dotenv import load_dotenv
from PIL import Image
import torch
from transformers import pipeline
import sys

load_dotenv()
# Check if Ollama is available
try:
   import ollama
   ollama.chat()
   print("Ollama is available.")
except ImportError:
   print("Ollama is not installed. Please install it to use this script.")
   exit()

# Load the resume and job description
def load_documents(master_resume_path, job_description_path):
    """Loads the master resume and job description from the given paths."""
    try:
        with open(master_resume_path, 'r', encoding='utf-8') as f:
            master_resume_text = f.read()
    except FileNotFoundError:
        print(f"Error: Master resume file not found at {master_resume_path}")
        sys.exit()
    try:
        with open(job_description_path, 'r', encoding='utf-8') as f:
            job_description_text = f.read()
    except FileNotFoundError:
        print(f"Error: Job description file not found at {job_description_path}")
        sys.exit()
    return master_resume_text, job_description_text

# Create a pipeline for text generation using Ollama
def create_ollama_pipeline(model_name="gemma3:4b"):
    """Creates an Ollama pipeline for text generation."""
    try:
        pipe = pipeline("text-generation", model=model_name, device=0)
        return pipe
    except Exception as e:
        print(f"Error creating Ollama pipeline: {e}")
        exit()

# Generate a tailored resume using Ollama
def generate_tailored_resume(master_resume_text, job_description_text, ollama_pipeline):
    """Generates a tailored resume using the Ollama model."""
    prompt = f"""
    You are an expert resume writer. Your task is to tailor the following resume to a specific job description.

    Here is the master resume:
    ```
    {master_resume_text}
    ```

    Here is the job description:
    ```
    {job_description_text}
    ```
    Please rewrite the resume to highlight the skills and experiences most relevant to the job description. Focus on using keywords from the job description and rephrasing accomplishments to demonstrate suitability for the role.
    Output the tailored resume in a clear and professional format.
    """
    try:
        response = ollama_pipeline(prompt, max_length=2000, temperature=0.7, top_p=0.95)
        tailored_resume_text = response[0]['generated_text']
        return tailored_resume_text
    except Exception as e:
        print(f"Error generating tailored resume: {e}")
        return None

# Save the tailored resume to a file
def save_resume(tailored_resume_text, output_path):
    """Saves the tailored resume to the specified output path."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tailored_resume_text)
        print(f"Tailored resume saved to {output_path}")
    except Exception as e:
        print(f"Error saving tailored resume: {e}")

if __name__ == "__main__":
    # Get file paths from the user



    # Paste the job description text here
    job_description_text = """
    We are seeking a highly motivated and experienced Software Engineer to join our growing team.
   The ideal candidate will have a strong understanding of software development principles and experience
   with Python, Java, and cloud technologies (AWS, Azure).
   Responsibilities include designing, developing, and testing software solutions,
   collaborating with other engineers, and contributing to the overall software architecture.
   Experience with Agile development methodologies is a plus.
   """

   master_resume_text = """
   John Doe
   (123) 456-7890 | john.doe@email.com | linkedin.com/in/johndoe

   Summary
   Highly motivated Software Engineer with 5+ years of experience in designing and developing software solutions. Proficient in Python, Java, and cloud technologies. Proven ability to work effectively in Agile development environments.

   Experience
   Software Engineer, ABC Company (2018-Present)
   - Developed and maintained web applications using Python and Django.
   - Designed and implemented RESTful APIs.
   - Collaborated with a team of engineers to deliver high-quality software.

   Education
   Bachelor of Science in Computer Science, University of California, Berkeley
   """
   output_path = "tailored_resume.txt"
    # Load the documents

   master_resume_text, job_description_text = load_documents("master_resume.txt", "job_description.txt")

    # Create the Ollama pipeline
    ollama_pipeline = create_ollama_pipeline()

    # Generate the tailored resume
    tailored_resume_text = generate_tailored_resume(master_resume_text, job_description_text, ollama_pipeline)

   # Save the tailored resume to a file
   save_resume(tailored_resume_text, output_path)

    if tailored_resume_text:
        # Determine output format
        if output_path.lower().endswith(".pdf"):
            try:
                # Attempt to convert text to PDF using Pillow (basic conversion)
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter

                c = canvas.Canvas(output_path, pagesize=letter)
                textobject = c.beginText()
                textobject.setTextOrigin(10, 730)
                textobject.setFont("Helvetica", 12)
                textobject.textLine(tailored_resume_text)
                c.drawText(textobject)
                c.save()
                print(f"Tailored resume saved as PDF to {output_path}")
            except ImportError:
                print("reportlab library not found. Saving as plain text.")
                save_resume(tailored_resume_text, output_path)
            except Exception as e:
                print(f"Error converting to PDF: {e}. Saving as plain text.")
                save_resume(tailored_resume_text, output_path)
        else:
            save_resume(tailored_resume_text, output_path)
```
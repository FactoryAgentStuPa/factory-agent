import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Laad je Gemini API-sleutel uit het .env bestand
load_dotenv()

gemini_llm = LLM(
model="gemini/gemini-3.5-flash-lite",
api_key=os.getenv("GEMINI_API_KEY")
)

# 1. De drie agents uit Fase 1
architect = Agent(
role="Software Architect",
goal="Ontwerp een eenvoudige structuur voor een rekenmachine in Python.",
backstory="Je bent een ervaren architect die houdt van simpele, duidelijke oplossingen.",
llm=gemini_llm,
verbose=True
)

developer = Agent(
role="Python Developer",
goal="Schrijf werkende Python-code op basis van het ontwerp van de architect.",
backstory="Je schrijft nette en foutloze Python-code.",
llm=gemini_llm,
verbose=True
)

tester = Agent(
role="QA Tester",
goal="Controleer de geschreven Python-code op fouten en geef een korte beoordeling.",
backstory="Je bent een kritische tester die elk detail nakijkt.",
llm=gemini_llm,
verbose=True
)

# 2. De opdrachten
taak_ontwerp = Task(
description="Maak een kort plan voor een Python-functie die twee getallen optelt en aftrekt.",
expected_output="Een korte opsomming van de benodigde functies.",
agent=architect
)

taak_code = Task(
description="Schrijf de Python-code volgens het plan van de architect.",
expected_output="Werkende Python-code.",
agent=developer
)

taak_test = Task(
description="Controleer de code van de developer en bevestig of deze veilig en correct is.",
expected_output="Een kort testrapport met goedkeuring.",
agent=tester
)

# 3. De Crew samenstellen en uitvoeren
demo_crew = Crew(
agents=[architect, developer, tester],
tasks=[taak_ontwerp, taak_code, taak_test],
verbose=True
)

resultaat = demo_crew.kickoff()
print("\n=== EINDRESULTAAT VAN DE CREW ===")
print(resultaat)

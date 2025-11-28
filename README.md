### Current state

1. RL
  
  https://www.youtube.com/watch?v=WsvFL-LjA6U&list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX

4. MIT 6.824: Distributed Systems:
   https://pdos.csail.mit.edu/6.824/schedule.html
   https://www.youtube.com/@6.824

   Lec 1 done out of 20

5. LLM
   https://www.youtube.com/watch?v=Ub3GoFaUcds

6. MCP

7. Non Convex Articel

8. RL
   why can't we using backpropogation on RL policy and end to end learning
   how is attention built in RL? Why is this hard attention?
   Is RL problem differentiable?

9. LLMs
   https://github.com/aishwaryanr/awesome-generative-ai-guide

   Upto and including week 2 done.

10. Amazon

11. Pytorch

12. Autograd lecture my Andre Kapathry
    https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
 
14. Causal Analysis

15. BedRock AWS

16. Langchain

17. Azure AI
 
18. AI lifecycle practices.


Formal training or certification on software engineering concepts and advanced applied experience
Hands-on practical experience delivering system design, application development, testing, and operational stability
Proficiency in automation and continuous delivery methods
Proficient in all aspects of the Software Development Life Cycle
Advanced understanding of agile methodologies such as CI/CD, Application Resiliency, and Security
Development lead experience: requirements capture, task decomposition, time and effort estimation, delivery planning, testing, user acceptance testing.
Deep understanding of KDB technology and Q language. At least 7 years of professional experience using KDB. With at least an additional 2 years as a Lead. 
Deep understanding of KDB+tick design and data organization, performance implications of different approaches.
Practical experience developing/running large datasets and optimizing query performance.
Practical experience scaling and load-balancing of KDB applications.
Practical experience building resilient and high-availability KDB applications.


Formal training or certification on software engineering concepts and proficient advanced experience.
Recent hands-on experience as a back-end software engineer, especially with customer-facing, LLM-powered microservices
Proficiency in Java and Python programming languages
Experience designing and implementing effective tests (unit, component, integration, end-to-end, performance)
Excellent written and verbal communication skills in English
Familiarity with advanced AI/ML concepts and protocols, such as Retrieval-Augmented Generation, agentic system architectures, and Model Context Protocol
Strong interest in building generative AI applications and tooling
Experience with cloud technologies, distributed systems, RESTful APIs, and web technologies
Understanding of event-based architecture, data streaming, and messaging frameworks
Proficiency in operating, supporting, and securing mission-critical software applications
Understanding of various data stores (relational, non-relational, vector)
Ability to mentor team members on coding practices, design principles, and implementation patterns
Ability to manage stakeholders and prioritize deliverables across multiple work streams
 

Preferred Qualifications, Capabilities, and Skills:

Background in STEM with exposure to productionising machine learning systems
Experience with MLOps tools and platforms (e.g., MLflow, Amazon SageMaker, Databricks, BentoML, Arize)
Proficiency in cloud-native microservices architecture
Hands-on experience with Amazon Web Services (AWS)
Previous experience as a Platform engineer
Experience working in highly regulated environments or industries
 

Experience with MLOps practices and tools for managing the machine learning lifecycle
Experience building and deploying Generative AI applications, including familiarity with LLMOps
Exposure to cloud platforms (such as AWS, GCP, Azure)

####
🗓️ 7-Day Intensive Prep Plan (3–5 hours/day)

DAY 1 — Understand the Domain + Expectations
🟦 Goal: Speak intelligently about KYC, onboarding, and where AI brings value.


Study:


What is KYC?


What is onboarding?


Key pain points: document extraction, entity resolution, risk scoring, workflow automation


Why graph DBs matter: customer relationships, ownership networks




Prepare 3 strong examples of “AI impact in operations.”


Try to articulate:

“In KYC, AI improves entity resolution using graph databases, reduces manual review using LLM-based extraction, and enhances risk scoring using model-driven rules.”

Deliverables for today:
✔️ A simple narrative: “How AI transforms onboarding.”
✔️ Example use cases: document classification, entity extraction, risk flag summarization, automation.
✔️ High-level architecture in your head.

DAY 2 — Python for Systems + API Skills
🟦 Goal: Be ready for the Python engineering interview.
Focus areas:


Python packaging / structure: src/, modules, logging


Writing clean, testable code


FastAPI fundamentals:


Routers, dependency injection, Pydantic models, error handling




Async patterns


ETL/data pipeline Python patterns


Practice:
Write a working FastAPI service with:


/predict


Loading a model


Logging


Error handling


Pydantic schemas


Unit tests


This can be done in 2–3 hours.

DAY 3 — End-to-End AI System Design
🟦 Goal: Prepare for the 60-min onsite whiteboarding.
Study:


How to design a production AI system (LLMs, ML models)


Data ingestion → ETL → feature store → model → deployment → monitoring


How to scale:


Load balancing


Caching


Retries & backoff


Rollbacks


Model registry + experiment tracking (MLflow)




Practice 2–3 whiteboard problems:


Design an LLM-based document extraction pipeline for onboarding.


Design an entity-resolution system using a graph database.


Design an AI-powered internal code assistant.


Focus on:


Data flow


Infra components


API boundaries


Security & compliance


Monitoring + failure modes


Architectural tradeoffs



DAY 4 — Infrastructure (Terraform / CI/CD / Monitoring)
🟦 Goal: Competently discuss infra without needing expert-level detail.
Learn enough Terraform to talk about:


Modules


Variables


State


Provisioning:


S3 buckets


ECS/EKS services


IAM roles


VPC basics




For CI/CD:


Understand GitHub Actions or Jenkins pipelines:


Lint → Test → Build → Security scan → Deploy




Blue/green deployments


Canary rollout


Monitoring:


Grafana basics (dashboards, alerts)


Prometheus metrics


ML monitoring (drift, data quality)



DAY 5 — Databricks + MLflow
🟦 Goal: Be able to speak to these during the system design interview.
Study:


MLflow:


Experiment tracking


Model registry


Deployment gateways




Databricks:


Delta tables


Spark clusters


ETL pipelines




Practice:


Explain how you’d build:


A scalable ETL pipeline


A model training workflow


A feature store


A batch inference job


A streaming inference job (if relevant)





DAY 6 — Behavioral & Leadership + Storytelling
🟦 Goal: Ace the hiring manager interview.
Prepare STAR stories for:


Leading cross-functional projects


Pushing back on stakeholders


Making architecture decisions


Driving innovation


Communicating tradeoffs


Working with diverse teams


Prepare answers for:


“Why applied AI vs pure ML modeling?”


“Tell me a time you invented a new solution.”


“How do you influence without authority?”


“How do you decide what to build vs buy?”


Have 3 polished, leadership-oriented stories.

DAY 7 — Full Mock Interviews + Weak Spot Fixes
🟦 Run this checklist:
Technical Screen
✓ Can you explain an ML system end-to-end?
✓ Can you explain REST API integration?
✓ Can you outline data pipelines?
System Design
✓ Can you design a scalable AI infra with diagrams?
✓ Do you articulate tradeoffs well?
Python / Engineering
✓ Can you write clean, modular code?
✓ Do you know async, typing, error handling?
Hiring Manager
✓ Do you sound like a leader?
✓ Do you speak at the right altitude (not too low-level, not too academic)?

🎯 High-Value Prep Materials (Shortlist)
Python


FastAPI crash course (2 hrs total)


Clean Code Python cheatsheet


System Design for ML


Chip Huyen: Designing ML Systems


AWS: ML system reference architectures (free)


Terraform


2-hour YouTube bootcamp


Learn basics: modules, state, providers, IAM


Monitoring


Prometheus/Grafana introduction


ML monitoring via EvidentlyAI (even if you don’t use it)


Domain


Explainability in KYC


AKYC (Automated KYC) landscape


Graph DB entity resolution examples



🧩 Want a personalized 7-day schedule?
If you want, I can generate a calendarized hour-by-hour plan based on:


Your current skill level (0–10 scale for each area)


Your available hours per day


What type of interview you find hardest


If you share your skill level in:


Python engineering


FastAPI / APIs


System design


Terraform / infra


Data engineering


ML Ops (MLflow, pipelines)


KYC domain knowledge


…I’ll customize the plan perfectly for your needs.
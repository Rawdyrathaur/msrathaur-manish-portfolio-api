---
title: "Projects"
type: "project"
url: "/"
---
# Projects Built by Manish Singh Rathaur

## Project: Tab Story

### What Tab Story Is
Tab Story is a Chrome browser extension built by Manish that intelligently organizes browser tabs using automatic intent detection and AI-powered summaries. It helps users manage dozens of open tabs without losing track of what they were working on.

### Key Features of Tab Story
Tab Story automatically groups browser tabs based on user intent such as job search, learning, or shopping. It uses Chrome's built-in AI capabilities to analyze open tabs and generate meaningful groups. It also provides AI-generated summaries of webpages to help users quickly understand content. The extension includes a timeline view to track browsing sessions by time and intent, and instant search across tab titles, notes, and intent labels. All AI processing happens locally on the device — no data leaves the browser.

### Technology Stack for Tab Story
Tab Story was built using JavaScript and CSS for the Chrome extension frontend and interface logic, with Python used for AI-related logic. It integrates directly with Chrome's built-in AI API for summarization and intent detection — making it fully local with no external API calls.

### Manish's Role on Tab Story
Manish developed Tab Story as a solo project. He designed the architecture, implemented the AI-powered tab organization logic, built the user interface, handled publishing to the Chrome Web Store, and is actively working on marketing and user growth.

### Tab Story Status and Links
Tab Story is live on the Chrome Web Store with real active users.
Chrome Web Store: https://chromewebstore.google.com/detail/tab-story/nhjglpjgddjcjafdabmepgalnaejnleb
GitHub: https://github.com/Rawdyrathaur/Tab_story

---

## Project: Carbon Pulse

### What Carbon Pulse Is
Carbon Pulse is a microservices-based carbon footprint tracker built by Manish. It calculates and tracks carbon emissions using IEA-compliant formulas across a distributed three-service architecture.

### Key Features of Carbon Pulse
Carbon Pulse consists of three independent services working together — a Spring Boot Java backend, a Node.js service, and a React frontend. The system uses circuit breaker patterns to handle service failures gracefully, ensuring reliability even when individual services are down. The calculation engine follows IEA (International Energy Agency) standards for accurate carbon footprint data. The system was validated across more than 50 test scenarios using JUnit.

### Technology Stack for Carbon Pulse
Carbon Pulse is built with Spring Boot and Java for the core backend, Node.js as a secondary service, React for the frontend, and Docker for containerization and orchestration. Testing is handled with JUnit.

### Manish's Role on Carbon Pulse
Manish designed the full microservices architecture, implemented the IEA-compliant calculation engine, set up the circuit breaker patterns, and wrote the test suite covering 50+ scenarios.

### Carbon Pulse Status
Carbon Pulse is a complete portfolio project. Live demo is not currently hosted but the full source code is available on GitHub: https://github.com/Rawdyrathaur/OmniSupport-AI

---

## Project: OmniSupport AI

### What OmniSupport AI Is
OmniSupport AI is a distributed AI-powered customer support platform built by Manish using a six-service Spring Boot microservices architecture. It demonstrates production-level distributed systems design with async communication, security, and service discovery.

### Key Features of OmniSupport AI
OmniSupport AI uses six independently deployed Spring Boot services orchestrated with Docker Compose. The system includes JWT-based security for authentication and authorization. Kafka handles asynchronous message pipelines between services. PostgreSQL serves as the database layer. Service discovery is managed with Eureka. The entire system is containerized and deployable via a single Docker Compose command.

### Technology Stack for OmniSupport AI
OmniSupport AI is built with Java, Spring Boot (multiple services), Apache Kafka, PostgreSQL, Docker, Docker Compose, Eureka for service discovery, and JWT for authentication.

### Manish's Role on OmniSupport AI
Manish designed and implemented all six services from scratch, set up the Kafka async pipeline, configured the Eureka service registry, implemented JWT security, and containerized the entire system with Docker Compose.

### OmniSupport AI Status
OmniSupport AI is a complete portfolio project showcasing Manish's distributed systems knowledge. Source code: https://github.com/Rawdyrathaur
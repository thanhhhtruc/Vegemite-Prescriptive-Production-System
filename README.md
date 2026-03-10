<p align="center">
  <img src="./public/Vegemite.webp" alt="Vegemite Prescriptive Production System" width="120">
</p>

<h1 align="center">Vegemite Prescriptive Production System</h1>

<p align="center">
  An AI-driven prescriptive analytics system designed to optimize production efficiency and quality control within the manufacturing process.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

---

## Overview

The Vegemite Prescriptive Production System is a comprehensive decision-support platform that leverages machine learning to provide actionable insights for production operators. Unlike traditional descriptive tools, this system focuses on prescriptive analytics—identifying the optimal parameters to achieve desired quality outcomes and minimize downtime.

## Key Features

- **Real-time Monitoring**: centralized dashboard for production KPIs, machine health, and operational status.
- **Quality Prediction**: Estimating product quality metrics based on current machine input parameters.
- **Set-Point Optimization**: Recommending optimal machine settings to maximize yield and ensure consistency.
- **Risk Assessment**: Proactive identification of potential downtime risks and sensor instability.
- **Data Visualization**: High-fidelity charts for trend analysis, set-point deviations, and comparative performance.
- **Operational Interface**: Dedicated controls for machine configuration and parameter adjustments.

## Project Structure

```bash
├── app/               # Next.js App Router (Routes, Layouts, and API endpoints)
├── components/        # Reusable UI components and feature-specific widgets
│   ├── ui/            # Base design system components
│   └── charts/        # Specialized visualization components
├── models/            # Machine Learning artifacts and inference logic
├── data/              # Static configurations and regional datasets
├── lib/               # Shared utilities, constants, and logging services
├── hooks/             # Custom React hooks for state and data fetching
└── public/            # Static assets and media files
```

## Implementation Status

The current iteration of the system includes:

1.  **Production Intelligence**: A responsive web interface built with Next.js and Tailwind CSS.
2.  **Machine Learning Integration**: Built-in support for Task 1 (Quality Prediction) and Task 2 (Prescriptive Recommendation) models.
3.  **Real-time Analytics**: Hooks and services for monitoring machine stability and downtime risks.
4.  **Audit Logging**: Persistent activity logs for tracking manual adjustments and system-generated recommendations.
5.  **Design System**: A scalable component architecture based on professional design principles.

## Getting Started

### Prerequisites

- Node.js 18.x or later
- Python 3.9 or later
- npm or pnpm

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Install Python requirements (if applicable):
   ```bash
   pip install -r requirements.txt
   ```

### Development

To start the development server:

```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

## Technology Stack

- **Frontend**: Next.js (App Router), React, TypeScript
- **Styling**: Tailwind CSS, Shadcn UI
- **Data Visualization**: Recharts
- **Backend/Logic**: Python, Scikit-Learn
- **State Management**: React Hooks, Context API
- **Icons**: Lucide React



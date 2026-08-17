# Vehicle Routing Problem (VRP) Solver

## Overview
This repository contains pure implementation of Branch and Bound algorithm used to solve Vehicle Routing Problem (VRP). 

## Features
- Basic Branch and Bound approach for CVRP.
- Supports standard VRP constraints

## Installation
1. Create a clone of the repository:
   ```
   git clone https://github.com/Personal-hulisanin/The-vehicle-routing-problem.git
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
- Run the python code to solve the VRP:
   ```
   python .\src\Branch_and_Bound\bnb_cvrp.py -d .\data\distances_demo.csv -D .\data\demands_demo.csv -c 10
   ```

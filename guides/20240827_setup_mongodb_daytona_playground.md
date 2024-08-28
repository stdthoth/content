---
title: "Setting Up a MongoDB Playground with Daytona"
description: "Learn how to setup mongoDB playground with Daytona and with it using the Terminal or the official MongoDB extension "
date: 2024-08-27
author: "Tata Shalom"
tags: ["daytona", "devcontainer", "mongoDB"]
---

# Setting Up a MongoDB Playground with Daytona

## Introduction
This guide will show you how to set up a MongoDB playground in Daytona. Thi

## TL;DR

- Prerequisites
-  Overviews of MongoDB, DevContainers and Daytona.
-  How to setup a DevContainer configuration for MongoDB.
- Creating the MongoDB playground in Daytona.
- Interacting with the playground using the Shell
- Using the MongoDB official extension.
- Common Issues and Troubleshooting.
- Conclusions.

### Prerequisites
- Docker download [here](https://docs.docker.com/get-started/get-docker/)
- An IDE like VS Code can be gotten [here](https://code.visualstudio.com/download) or JetBrains [here](https://www.jetbrains.com/idea/download/)
- Daytona can be installed from [here](https://www.daytona.io/docs/installation/installation/)

## Overview of MongoDB 
MongoDB is a popular NoSQL database that is designed for storing and retrieving large amounts of unstructured or semi-structured data. Unlike traditional relational databases (SQL databases) that store data in tables with fixed schemas (rows and columns), MongoDB stores data in a more flexible, JSON-like format called BSON (Binary JSON). You can find out more about MongoDB and its core features [here](https://docs.mongodb.com/)

## Overview of DevContainer
A DevContainer is a development environment defined by a configuration file called **devcontainer.json** that uses Docker to create a consistent and reproducible workspace with all necessary dependencies, tools, and settings. It's primarily used with Visual Studio Code's Remote - Containers extension to provide portable, containerized development setups across different machines and teams. You can learn more about Dev Containers [here](https://www.containeers.dev/).

## Overview of Daytona
Daytona is a platform that simplifies the development environment setup by offering cloud-based dev environments for software developers. It enables users to create and share fully configured development environments in the cloud, allowing for faster onboarding, collaboration, and consistent setups across teams.

**Features of Daytona**
- **Pre-configured Environments**: You can create environments with all dependencies, tools, and configurations pre-installed, so developers can start coding immediately without having to spend time configuring their setups.
- **Collaborative Workspaces**:The platform enables team collaboration by allowing multiple developers to work on the same environment. This can be particularly useful for pair programming, code reviews, or troubleshooting.
- **Containerized Environments**:Each development environment can be containerized, ensuring that the setup is consistent, reproducible, and isolated from other environments. This helps avoid the common "works on my machine" problem.

For more information about Daytona check out its [docs](https://daytona.io/docs)


## Step 1: Setup a DevContainer Configuration for MongoDB

Here, you're going to create a dev container using a `devcontainer.json` file, a `Dockerfile` and a `docker-compose.yml` file.

- **Step 1.1**: Create a new directory

    You will create a file directory with any name and move inside of it.

    ```bash
    mkdir daytona-mongodb-playground && cd daytona-mongodb-playground
    ```
- **Step 1.2**: Create a .devcontainer directory

    This is where your devcontainer.json and dockerfiles will live

    ```bash
    mkdir .devcontainer && cd .devcontainer
    ```

- **Step 1.3**: Create a `devcontainer.json` file
    
    You are going to create a `devcontainer.json` file with the following code. This is the configuration file for the dev environment specifying settings and dependencies.
    ```json
    {
    "name": "mongo Dev Container Playground",
    "dockerComposeFile": "docker-compose.yml",
	"service": "mongodb_playground",
    "workspaceFolder": "/workspaces/${localWorkspaceFolderBasename}",
    "forwardPorts": [27017],
    "customizations": {
		"vscode": {
			// Add the IDs of extensions you want installed when the container is created.
			"extensions": [
				"mongodb.mongodb-vscode"
			]
		}
	}
  }
  
    ```
    Let's breakdown the `devcontainer.json` file 
    - **name**: Specifies the name of the development environment.
    - **dockerComposeFile**: points to the `docker-compose.yml` file we are going to use
    - **service**: Refers to the service name in the `docker-compose.yml` file that represents the MongoDB container or environment to be used.
    - **workspaceFolder**:Maps the local workspace folder (where the code is stored) to the /`workspaces/` directory inside the container. The variable `${localWorkspaceFolderBasename}` automatically resolves to the name of the local workspace folder.
    - **forwardPorts**: Exposes MongoDB's default port (`27017`) so that it is accessible from the host system or other containers for interaction with the MongoDB service.
    - **customizations**:Allows customization of the development environment, specifically for VS Code.
    - **vscode**: Automatically installs the MongoDB extension for VS Code, enabling MongoDB-related features (e.g., interacting with MongoDB directly from VS Code).
     Declares a persistent volume for storing MongoDB data.
- **Step 1.4**: Create a `Dockerfile`
    
    You will create a `Dockerfile` in the same directory

    ```dockerfile
    FROM mcr.microsoft.com/devcontainers/base:focal
    RUN  sudo apt-get install -y gnupg wget curl
    RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
    RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
    RUN sudo apt-gUpdates the package list and installs MongoDB.et update && export DEBIAN_FRONTEND=noninteractive \
    && sudo apt-get install -y mongodb-org
    ```
    Let's breakdown this Dockerfile for you
    - `FROM mcr.microsoft.com/devcontainers/base:focal`:Uses the Microsoft DevContainers base image based on Ubuntu 20.04 LTS (Focal Fossa). This provides a standard environment for development containers.
    - `RUN sudo apt-get install -y gnupg wget curl`:These utilities are necessary for managing GPG keys and downloading files. `gnupg` is used for key management, while `wget` and `curl` are used for downloading files.
    - `RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -`: Ensures that the packages from the MongoDB repository can be verified and trusted. The `-qO -` option makes `wget` output the file to stdout, which is then piped to `apt-key add` for key addition.
    - `RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list`: Configures the system to use MongoDB’s official repository for version 4.4. This allows `apt-get` to retrieve MongoDB packages for installation.
    - `RUN sudo apt-get update && export DEBIAN_FRONTEND=noninteractive && sudo apt-get install -y mongodb-org`: Updates the package list and installs MongoDB.

- **Step 1.5** Create a docker-compose.yml file:

    You are going to create `docker-compose.yml` file in the same directory

    ```yml
    version: '3.8'

    services:
        mongodb_playground:
            build:
                context: .
                dockerfile: Dockerfile
                volumes:
                    -../..:/workspaces:cached
                network_mode: service:db
                command: sleep infinity

        db:
            image: mongo:4.4
            restart: unless-stopped
            ports:
            - "27017:27017"
            volumes:
            - mongodb-data:/data/db
    volumes:
         mongodb-data:
    ```
    Let's breakdown the ``docker-compose.yml` file for you
    
    - **version**: Ensures compatibility with Docker Compose features and syntax defined in version 3.8.
    - **services**: Defines the containers (services) that will be run as part of this Compose setup.
    - **mongodb_playground**: Defines a service named `mongodb_playground`.
    - **context**: Sets the build context to the current directory.
    - **dockerfile**: Specifies the Dockerfile to use for building this service.
    - **volumes: ../..:/workspaces:cached**: Mounts a directory from the host (two levels up to /workspaces in the container. The cached option optimizes performance by indicating that changes in the host are less likely to affect the container.

    - **network_mode: service:db**: Allows mongodb_playground to communicate with the db container using the same network. This is useful for sharing network settings between containers.
    - **command: sleep infinity**: Keeps the container running indefinitely, preventing it from stopping after its initial command finishes. 
    - **db**: Defines a service named `db`.
    - **image: mongo:4.4**:Uses the official MongoDB image version 4.4.
    - **restart: unless-stopped**: Restarts the container unless it is explicitly stopped by the user. This helps ensure the MongoDB service remains available.
    - **ports**: Exposes MongoDB’s default port for external access. This makes MongoDB accessible from the host machine.
    - **volumes**:
        - `mongodb-data:/data/db`: Mounts a named volume `mongodb-data` to /`data/db` in the container.

    - **volumes**: This volume is used by the db service to store its database files.

- **Step 1.6** Initialize,Commmit and Create a GitHub repository:
    You will initialize and commit the code in your current directory 

    ```bash
    git init
    git add .
    git commit -m "inital commit"
    ```
    After commiting your code you will push it to a remote repository of your choice.

    ```bash
    git remote add origin https://github.com/YOUR-GITHUB-USERNAME/YOUR-DIRECTORY-NAME.git
    git branch -M main
    git push -u origin main
  ```
  You can find the Github repository where my devcontainer files which i used for this guide [here](https://github.com/stdthoth/daytona-mongodb-playground).

## Step 2 Creating the MongoDB playground in Daytona

Here you are going to use Daytona to build the playground using Github as a Provider and open a workspace in VS Code. You should ensure `daytona` is installed on you machine before proceeding.

- **Step 2.1** Start the Daytona Server daemon :

    Execute the command provided below to start the `daytona` server daemon. when prompted to start the server in the current terminal session click `yes`
    ```bash
  daytona server
  ```
    

    Your output should look like the image below

    ![image of running daytona server](assets/20240827_setup_mongodb_daytona_playground_img_1.png) the container.
    
- **Step 2.1** Setup your Git Provider :
    Execute the command provided below to add your git provider

    ```bash
    daytona git-provider add
    ```
    Your output should be similar to the image below
    
    ![image of github provider](assets/20240827_setup_mongodb_daytona_playground_img_2.png)
    
    Select GitHub and provide your personal access token.

- **Step 2.2** Choose your preferred IDE :
     
    Run this command in terminal to choose your  IDE.
     ```bash
    daytona ide
    ```
- **Step 2.3** Create a Daytona Workspace :
    
    Substitute the `USERNAME/REPO-NAME` from the code below with the Github username and the repository where the devcontainer is stored. 

  ```bash
  daytona create https://github.com/USERNAME/REPONAME
  ```
  Ideally but optional you should add the `--code` flag at the end. This will automatically open your preferred editor after building the devcontainer

  ```bash
  daytona create https://github.com/USERNAME/REPONAME --code
  ```
    Your output before your IDE opens should be similar to these images

    ![image of creating dayona workspace](assets/20240827_setup_mongodb_daytona_playground_img_3.png)

    ![image of creating daytona workspace_2](assets/20240827_setup_mongodb_daytona_playground_img_4.png)

    ![image of creating daytona workspace_3](assets/20240827_setup_mongodb_daytona_playground_img_5.png)
   

# Step 3 Interacting with the MongoDB playground using Mongo Shell.

Your Preferred IDE should have opened up, you will be instructed to open a Remote SSH connection.You should see that your `devcontainer.json`, `Dockerfile` and `docker-compose.yml` files have been downloaded.Navigate to your terminal after creating the connection by clicking `Ctrl + Shift + ` ` . In the terminal execute the following code to start a mongo shell and carry out operations in .

    ```bash
        mongo
    ```
    
- **Step 3.2** Create a MongoDB Database: 

    ```bash
        use test
    ```


- **Step 3.2** Create MongoDB Collection:

     Create a new Mongo Collection
Creating the MongoDB playground in Daytona
     ```bash
     db.createCollection("collection")
    ```

- **Step 3.3**: Create a document into test collection:

  ```bash
  db.collection.insertOne({ name: "John", age: 25, city: "New York" });
  ```
  Your output should look like the image below
  
  ![image of creating a document](assets/20240827_setup_mongodb_daytona_playground_img_6.png)

- **Step 3.4**: Read (Query) a document from the collection:
    ```bash
    db.collection.findOne({ name: "John" })
    ```
    Your output should look like the image below

    ![image of reading a document](assets/20240827_setup_mongodb_daytona_playground_img_7.png)

- **Step 3.5**: Update a document:
    ```bash
    db.collection.updateOne(
    { name: "John" },  // Filter
    { $set: { age: 26 } }  // Update operation
    );
    ```
    Your output should look like the image below
  
  ![image of updating a document](assets/20240827_setup_mongodb_daytona_playground_img_8.png)

- **Step 3.6**: Delete the Collection
    ```bash
    db.collection.deleteOne({ name: "John" });
    ```
    Your output should look like the image below
    
    ![image of deleting a document](assets/20240827_setup_mongodb_daytona_playground_img_9.png)

## Step 4 Using the MongoDB playground extension (Optional)

Alternatively you can use the official MongoDB extensions playground since you provided `"mongodb.mongodb-vscode"` as a extension in the `devcontainer.json` file.

- **Step 4.1** Create a connection:

    Click on the **Connect to MongoDB** in the MongoDB view, then enter the connection string below 

    ```bash
    mongodb://localhost:27017
    ```
- **Step 4.2** Connect to Playground:

    Connect to the MongoDB playground, which is a built-in interactive environment for running MongoDB queries. Navigate to the MongoDB extension and click on the **Create MongoDB Playground**. 

- **Step 4.3** Run Queries :
    Inside the playground you can write queries just like you did in the MongoDB Shell. For example

    ```javascript
    use("test");
    db.collection.insertOne({name:'Eliza',age:40})
    ```
    You should get the same output below
    
    ![image of mongodb vscode extension](assets/20240827_setup_mongodb_daytona_playground_img_10.png)

## Common Issues and Troubleshooting

**Connection Refused on Port 3896**: If you are experiencing this, check what service is running on that port and close it. Check out how to find and close a port connection [here](https://www.cyberciti.biz/faq/unix-linux-check-if-port-is-in-use-command/)

**Cannot Start the Daytona Docker Registry**: If you cant start the Daytona Docker registry don't panic, just re-run the `daytona create` command and it should run as expected.

**IDE doesn't open after building Workspace** : If your IDE doesnt start immediately after building the devcontainer, open your terminal and type the `daytona list` command,select your running workspace and you're good to go.

## Conclusion
After following this guide you should be comfortable setting up your own devcontainer configuration and running on Daytona. From here, you should explore other features of Daytona such as the Registry, Prebuilds e.t.c

## References

- [MongoDB Documentation](https://docs.mongodb.com/)

- [Daytona Documentation](https://daytona.io/docs)

- [DevContainer Specification](https://containers.dev/)
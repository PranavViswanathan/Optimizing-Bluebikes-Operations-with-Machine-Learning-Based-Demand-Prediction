# Use a lightweight Node.js runtime
FROM node:18-alpine

# Create app directory
WORKDIR /app

# Copy only package files first (for cached layer)
COPY package*.json ./

# Install dependencies
RUN npm install --production

# Copy the rest of the backend code
COPY . .

# Expose port (Cloud Run ignores but good practice)
EXPOSE 8080

# Start the backend
CMD ["npm", "run", "start"]

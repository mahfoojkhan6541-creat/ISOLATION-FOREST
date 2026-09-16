FROM node:22-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-numpy python3-pandas python3-openpyxl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN npm install -g corepack@latest \
    && corepack pnpm install \
    && corepack pnpm run build

ENV NODE_ENV=production
ENV PYTHON_BIN=python3
CMD ["node", "dist/index.js"]

FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# The nginx entrypoint runs envsubst over /etc/nginx/templates at start-up, so
# the API key is read from the environment instead of baked into the image.
# The filter keeps substitution to ATLAS_* and leaves nginx's own $uri alone.
COPY templates/default.conf.template /etc/nginx/templates/default.conf.template
ENV NGINX_ENVSUBST_FILTER=^ATLAS_
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80

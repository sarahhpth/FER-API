# FER-API
This repository includes a ready-to-use deployable facial emotion recognition model and its API.

Request: 

![Request body](img/request.png)

Response: 

![Response body](img/response.png)

To build using Docker:
`docker build -t fer-api .`

To run using Docker (locally):
`docker run -p 8000:80 fer-api`

# Experiment 8: Publish Docker Image to GitHub Container Registry (GHCR)

## Aim
To publish the tested Employee Promotion ML Docker image to GitHub Container Registry (GHCR), pull the image from the registry, and verify that the application runs successfully.

## Docker Image Details
- **GHCR Image:** `ghcr.io/vasanthi286/employee-promotion-ml:v1`
- **Registry:** GitHub Container Registry
- **Package Visibility:** Private
- **Application:** Employee Promotion Prediction API
- **Port:** 5000

## Procedure

### 1. Login to GHCR
```powershell
docker login ghcr.io -u Vasanthi286 --password-stdin
```

### 2. Publish the Docker Image
```powershell
docker push ghcr.io/vasanthi286/employee-promotion-ml:v1
```

### 3. Verify the Package
The `employee-promotion-ml` package and `v1` tag were verified on GitHub Packages.

### 4. Pull the Published Image
```powershell
docker pull ghcr.io/vasanthi286/employee-promotion-ml:v1
```

### 5. Run the Container
```powershell
docker run -d --name employee-promotion-ghcr-container -p 5000:5000 ghcr.io/vasanthi286/employee-promotion-ml:v1
```

### 6. Verify the Container
```powershell
docker ps
docker logs employee-promotion-ghcr-container
```

### 7. Test the Prediction API
The `/predict` endpoint was tested using a POST request with JSON employee data.

Endpoint: `http://localhost:5000/predict`

Response:
```json
{
  "promotion_prediction": 0
}
```

## Results
- Docker image successfully pushed to GHCR.
- Package and `v1` tag verified on GitHub.
- Published image successfully pulled from GHCR.
- Container started and ran on port 5000.
- Homepage returned HTTP 200.
- Prediction API returned `promotion_prediction: 0` for the sample input.

## Conclusion
The Employee Promotion ML Docker image was successfully published to GitHub Container Registry, retrieved from the registry, and run in a new container. The homepage and prediction API were tested successfully.


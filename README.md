# Photo Finder on Coolify

Guests upload a selfie and get every photo they appear in. Free, runs on your own VPS.

## 1. Put the project in a Git repo

Create a free **private** GitHub repo and push this folder to it (photos are not part of it).
Coolify needs GitHub access: in Coolify go to **Sources** and connect GitHub, or use a deploy key.

## 2. Create the app in Coolify

1. **Project > New Resource > Private Repository** (or Public if you made it public), pick the repo.
2. Build Pack: **Dockerfile**. Ports Exposes: **3000**.
3. Domains: Coolify can fill in a free `*.sslip.io` address for your server IP, or enter your own domain
   (with an A record pointing to the server). Coolify gets the HTTPS certificate automatically.
4. **Persistent Storage > Add > Directory Mount**:
   - Source path on the server: `/opt/photo-finder-storage`
   - Destination path in the container: `/storage`
5. Under **Advanced**, tick "Enable Gzip", leave everything else as is. Press **Deploy**.

The first build takes a few minutes. The site opens now but shows "Photos are still being prepared"
until you finish step 4 below.

## 3. Copy your Google Drive photos onto the server

SSH into the VPS (this is the host, not a container):

```bash
apt update && apt install -y rclone
mkdir -p /opt/photo-finder-storage/photos
rclone config
```
In `rclone config`: `n` (new remote), name `gdrive`, type `drive`, accept defaults. For "Use web browser
to automatically authenticate?" answer `n`, run the command it prints on your own computer (install rclone
there too), and paste the code back. Then:

```bash
rclone copy "gdrive:My Event Photos" /opt/photo-finder-storage/photos --progress
```

## 4. Index the photos (30 to 90 minutes, once)

Still on the VPS:

```bash
docker ps --format '{{.Names}}' | grep -v coolify     # find your app's container name
docker exec -d YOUR_CONTAINER_NAME sh -c "python index_photos.py > /storage/index.log 2>&1"
tail -f /opt/photo-finder-storage/index.log
```

The first run downloads the face model (about 300 MB) into the storage folder. When the log says
"Done", the site starts matching right away, with no restart needed. If you stop it, run the same
command again and it resumes. If you redeploy while it runs, start it again.

## Tuning

Set these in Coolify under **Environment Variables**, then redeploy:

- `MATCH_THRESHOLD=0.45` if wrong people appear (stricter); `0.35` if people are missing (looser). Default `0.40`.
- For missed small faces in big group photos, run indexing with a larger size:
  `docker exec -d NAME sh -c "DET_SIZE=1280 python index_photos.py > /storage/index.log 2>&1"` (slower).
- Edit the headline and text in `static/index.html`, push to Git, and Coolify redeploys.

## Privacy

Selfies are processed in memory and never saved. Share the link only with people who attended.

# Optional account sign-in

The dashboard uses Streamlit's OpenID Connect sign-in. **Log in** at the bottom-right opens the sign-in pop-up; the same option is under **Settings & account** in the sidebar. Sign-in is optional, and the dashboard stays public when nobody is signed in.

## 1. Create a Google sign-in client

1. Open [Google Auth Platform → Branding](https://console.cloud.google.com/auth/branding) and select or create a Google Cloud project.
2. Set the app name to **Earth Emergency Command** and choose a support email.
3. Open **Audience**. For a personal app, choose **External** and add your Google account under **Test users**. For a Google Workspace app restricted to your organization, choose **Internal** if that option is available.
4. Open **Clients → Create client → Web application**.
5. Under **Authorized redirect URIs**, add this exact URL:

   `https://testnasa123.streamlit.app/oauth2callback`

6. Create the client. Copy its **Client ID** and **Client secret**. Keep the secret private; never commit it to GitHub or paste it into chat.

While an External Google app is in **Testing**, only listed test users can sign in, and test-user grants expire after seven days. That is fine for setup and initial testing.

## 2. Add the credentials to Streamlit Cloud

1. Open the app's [Streamlit Cloud page](https://testnasa123.streamlit.app/), click **Manage app**, then open **Settings → Secrets**.
2. Add the following TOML, replacing the four placeholders. Generate a fresh cookie secret locally; for example, run `python -c "import secrets; print(secrets.token_urlsafe(32))"` in PowerShell and copy its output into `cookie_secret`.

```toml
[auth]
redirect_uri = "https://testnasa123.streamlit.app/oauth2callback"
cookie_secret = "PASTE_YOUR_NEW_RANDOM_VALUE_HERE"
client_id = "PASTE_GOOGLE_CLIENT_ID_HERE"
client_secret = "PASTE_GOOGLE_CLIENT_SECRET_HERE"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

3. Save the secrets and wait for Streamlit to restart the app.
4. Click the bottom-right **Log in** button and sign in with a listed test account. The dashboard should return with an **Account** button.

Do not add these values to the repository. The client secret is stored only in Streamlit Cloud Secrets. Local development needs a separate `[auth]` configuration with `redirect_uri = "http://localhost:8501/oauth2callback"` and that localhost URI must also be authorized in Google Cloud.

This sign-in establishes user identity only. It does not currently save dashboard preferences between sessions.

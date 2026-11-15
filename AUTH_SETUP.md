# Optional account sign-in

## AQI and fire email alerts

Email delivery uses Resend SMTP. Create a Resend API key and add a domain you own in Resend. Copy the DNS records Resend provides into your domain provider and wait for verification. Use a sender address at that verified domain. [Resend SMTP settings](https://resend.com/docs/send-with-smtp), [create an API key](https://resend.com/docs/create-an-api-key), and [verify a domain](https://resend.com/docs/add-a-domain).

```toml
[smtp]
SMTP_HOST = "smtp.resend.com"
SMTP_PORT = 465
SMTP_USER = "resend"
SMTP_PASSWORD = "re_PASTE_RESEND_API_KEY_HERE"
SMTP_FROM = "Earth Emergency Command <alerts@YOUR_VERIFIED_DOMAIN>"
SMTP_USE_SSL = true
```

Add this block in Streamlit Cloud → **Manage app → Settings → Secrets**. Resend uses `resend` as the SMTP username and the API key as the SMTP password; port 465 uses implicit SSL/TLS. Keep the API key private and out of GitHub. In the dashboard, send a test AQI email or test fire email. AQI email alerts are sent when a watched place reaches the threshold during a comparison or while Auto refresh is active. New-fire emails are sent after the initial live FIRMS snapshot, when a later refresh finds new detections.

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

## Phone-number sign-in

Phone sign-in uses Twilio Verify to send and validate a one-time SMS code. A verified phone sign-in lasts for the current Streamlit browser session. The app does not store a phone account or preserve sign-in across sessions.

1. In Twilio, create a **Verify Service** and copy its Service SID (starts with `VA`).
2. Add the credentials to Streamlit Cloud → **Manage app → Settings → Secrets**. Keep the client secrets private. The `[twilio]` table is optional, but these values must be together under it if you use this format:

```toml
[twilio]
TWILIO_ACCOUNT_SID = "AC..."
TWILIO_AUTH_TOKEN = "..."
TWILIO_VERIFY_SERVICE_SID = "VA..."
```

Twilio settings are only used when someone chooses phone-number sign-in. No Twilio messaging sender is needed for dashboard alerts. Phone numbers entered in the sign-in form must use international format, such as `+15551234567`. Twilio Verify validates the number before the app marks that browser session as signed in.

Fire and AQI alerts are delivered by email. AQI reminders are checked while the dashboard is open with Auto refresh enabled.

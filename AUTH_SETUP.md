# Optional account sign-in

The dashboard uses Streamlit's OpenID Connect sign-in. The bottom-right **Log in** button opens the sign-in pop-up; the same action is available under **Settings & account** in the sidebar.

To enable real sign-in on Streamlit Community Cloud:

1. Create an OAuth/OIDC client with your identity provider (the in-app example uses Google).
2. Add this configuration in the app's **Manage app → Settings → Secrets**. Replace the placeholders with the provider values and a long, random cookie secret.

```toml
[auth]
redirect_uri = "https://testnasa123.streamlit.app/oauth2callback"
cookie_secret = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
client_id = "REPLACE_WITH_PROVIDER_CLIENT_ID"
client_secret = "REPLACE_WITH_PROVIDER_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

3. Register `https://testnasa123.streamlit.app/oauth2callback` as an allowed redirect URL with the identity provider.
4. Reboot the app. Signing in is optional; the dashboard remains public when nobody is signed in.

For local development, use `http://localhost:8501/oauth2callback` as the redirect URI and allow it in the provider's client settings.

This sign-in establishes a user's identity. It does not by itself save personalized dashboard preferences between sessions.

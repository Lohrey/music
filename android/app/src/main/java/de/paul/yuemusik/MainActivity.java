package de.paul.yuemusik;

import android.app.Activity;
import android.app.DownloadManager;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.CookieManager;
import android.webkit.URLUtil;
import android.webkit.ValueCallback;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.Manifest;
import android.content.pm.PackageManager;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

/**
 * "Musik machen" — a full-screen window onto the YuE2 music server on Paul's PC.
 * First start asks once for the server address; after that it just opens.
 */
public class MainActivity extends Activity {
    private static final int BG = Color.rgb(14, 14, 18);
    private static final int ACCENT = Color.rgb(255, 92, 138);
    private static final int FILE_REQUEST = 42;
    private static final int MIC_REQUEST = 43;
    private PermissionRequest pendingMic;

    private SharedPreferences prefs;
    private FrameLayout root;
    private WebView web;
    private ProgressBar spinner;
    private ValueCallback<Uri[]> fileCallback;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        prefs = getSharedPreferences("app", MODE_PRIVATE);
        root = new FrameLayout(this);
        root.setBackgroundColor(BG);
        setContentView(root);
        String url = prefs.getString("url", "");
        if (url.isEmpty()) showSetup(null);
        else showWeb(url);
    }

    // ───────────────────────── setup screen ─────────────────────────
    private void showSetup(String problem) {
        if (web != null) { web.destroy(); web = null; }
        root.removeAllViews();
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setGravity(Gravity.CENTER_HORIZONTAL);
        int pad = dp(28);
        box.setPadding(pad, dp(90), pad, pad);

        TextView title = text("🎵 Musik machen", 30, true);
        box.addView(title);
        TextView sub = text(problem != null ? problem
                : "Einmal die Adresse deines Musik-Servers eintragen.\nDanach öffnet sich die App direkt.", 16, false);
        sub.setTextColor(problem != null ? Color.rgb(255, 170, 120) : Color.rgb(154, 152, 163));
        sub.setPadding(0, dp(10), 0, dp(24));
        box.addView(sub);

        EditText input = new EditText(this);
        input.setHint("z. B. meinname.ngrok-free.app");
        input.setText(prefs.getString("url", ""));
        input.setTextColor(Color.WHITE);
        input.setHintTextColor(Color.rgb(110, 108, 118));
        input.setTextSize(18);
        input.setSingleLine(true);
        input.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        input.setPadding(dp(16), dp(14), dp(16), dp(14));
        input.setBackground(rounded(Color.rgb(31, 31, 39), dp(14)));
        box.addView(input, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        Button go = bigButton(problem != null ? "Nochmal versuchen" : "Los geht's");
        go.setOnClickListener(v -> {
            String u = normalize(input.getText().toString());
            if (u == null) { Toast.makeText(this, "Bitte eine Adresse eintragen", Toast.LENGTH_SHORT).show(); return; }
            prefs.edit().putString("url", u).apply();
            showWeb(u);
        });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(60));
        lp.topMargin = dp(20);
        box.addView(go, lp);
        root.addView(box);
    }

    private static String normalize(String raw) {
        String u = raw.trim();
        if (u.isEmpty()) return null;
        if (!u.startsWith("http://") && !u.startsWith("https://")) u = "https://" + u;
        while (u.endsWith("/")) u = u.substring(0, u.length() - 1);
        return u + "/";
    }

    // ───────────────────────── the web app ─────────────────────────
    @SuppressWarnings("SetJavaScriptEnabled")
    private void showWeb(String url) {
        root.removeAllViews();
        web = new WebView(this);
        web.setBackgroundColor(BG);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setAllowFileAccess(false);
        s.setUserAgentString("YuEMusikApp/1.0 (Linux; Android) Mobile");
        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(web, true);

        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String u) {
                spinner.setVisibility(View.GONE);
                CookieManager.getInstance().flush();
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest req, WebResourceError err) {
                if (req.isForMainFrame()) {
                    showSetup("Der Musik-Server ist gerade nicht erreichbar.\nLäuft der PC und ist Pinokio gestartet?");
                }
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                Uri target = req.getUrl();
                Uri home = Uri.parse(prefs.getString("url", ""));
                if (target.getHost() != null && target.getHost().equals(home.getHost())) return false;
                startActivity(new Intent(Intent.ACTION_VIEW, target));
                return true;
            }
        });

        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> {
                    boolean wantsMic = false;
                    for (String r : request.getResources()) {
                        if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(r)) wantsMic = true;
                    }
                    if (!wantsMic) { request.deny(); return; }
                    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                        request.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
                    } else {
                        pendingMic = request;
                        requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, MIC_REQUEST);
                    }
                });
            }

            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = callback;
                Intent pick = new Intent(Intent.ACTION_GET_CONTENT);
                pick.addCategory(Intent.CATEGORY_OPENABLE);
                pick.setType("audio/*");
                try {
                    startActivityForResult(Intent.createChooser(pick, "Musikdatei wählen"), FILE_REQUEST);
                } catch (Exception e) {
                    fileCallback = null;
                    return false;
                }
                return true;
            }
        });

        web.setDownloadListener((dlUrl, userAgent, disposition, mime, length) -> {
            try {
                DownloadManager.Request r = new DownloadManager.Request(Uri.parse(dlUrl));
                String cookie = CookieManager.getInstance().getCookie(dlUrl);
                if (cookie != null) r.addRequestHeader("Cookie", cookie);
                r.addRequestHeader("User-Agent", userAgent);
                String name = URLUtil.guessFileName(dlUrl, disposition, mime);
                r.setTitle(name);
                r.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
                r.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, name);
                ((DownloadManager) getSystemService(DOWNLOAD_SERVICE)).enqueue(r);
                Toast.makeText(this, "Song wird gespeichert (Ordner Downloads)", Toast.LENGTH_LONG).show();
            } catch (Exception e) {
                startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(dlUrl)));
            }
        });

        root.addView(web, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        spinner = new ProgressBar(this);
        FrameLayout.LayoutParams sp = new FrameLayout.LayoutParams(dp(56), dp(56), Gravity.CENTER);
        root.addView(spinner, sp);

        // small "address" button (bottom-left) to change the server later
        TextView gear = text("⚙", 18, false);
        gear.setTextColor(Color.rgb(90, 88, 98));
        gear.setPadding(dp(12), dp(8), dp(12), dp(8));
        gear.setOnLongClickListener(v -> { showSetup(null); return true; });
        gear.setOnClickListener(v -> Toast.makeText(this, "Gedrückt halten, um die Server-Adresse zu ändern", Toast.LENGTH_SHORT).show());
        root.addView(gear, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM | Gravity.START));

        web.loadUrl(url);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        if (request == FILE_REQUEST && fileCallback != null) {
            Uri[] picked = null;
            if (result == RESULT_OK && data != null && data.getData() != null) picked = new Uri[]{data.getData()};
            fileCallback.onReceiveValue(picked);
            fileCallback = null;
            return;
        }
        super.onActivityResult(request, result, data);
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        if (code == MIC_REQUEST && pendingMic != null) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                pendingMic.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
            } else {
                pendingMic.deny();
                Toast.makeText(this, "Ohne Mikrofon-Erlaubnis geht Vorsingen nicht", Toast.LENGTH_LONG).show();
            }
            pendingMic = null;
            return;
        }
        super.onRequestPermissionsResult(code, perms, results);
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onPause() {
        super.onPause();
        CookieManager.getInstance().flush();
    }

    // ───────────────────────── helpers ─────────────────────────
    private int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    private TextView text(String t, int size, boolean bold) {
        TextView v = new TextView(this);
        v.setText(t);
        v.setTextSize(size);
        v.setTextColor(Color.rgb(243, 241, 238));
        v.setGravity(Gravity.CENTER);
        if (bold) v.setTypeface(Typeface.DEFAULT_BOLD);
        return v;
    }

    private Button bigButton(String label) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(20);
        b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setTextColor(Color.WHITE);
        GradientDrawable g = new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{ACCENT, Color.rgb(255, 154, 60)});
        g.setCornerRadius(dp(30));
        b.setBackground(g);
        return b;
    }

    private static GradientDrawable rounded(int color, int radius) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(color);
        g.setCornerRadius(radius);
        return g;
    }
}

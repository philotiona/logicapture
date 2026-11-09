"""Shared visual hierarchy for the three purposeful workspaces."""
CSS = """<style>
.stApp {background:#fff;color:#182234}
.block-container {max-width:1100px;padding:24px 24px 48px}
header[data-testid="stHeader"],footer {display:none}
html,body,[data-testid="stAppViewContainer"],button,input,textarea {
 font-family:'Segoe UI',Arial,sans-serif !important}
h1,h2,h3,h4 {font-family:'Segoe UI',Arial,sans-serif !important;color:#182234 !important;
 font-weight:650 !important;letter-spacing:-.025em}
h2 {font-size:1.7rem !important} h3 {font-size:1.25rem !important}
p,li {font-size:.96rem;line-height:1.55}
button,input,textarea,[data-baseweb="select"]>div,details {border-radius:8px !important}
button {font-size:.93rem !important}
button[kind="primary"]:not(:disabled) {background:#002FA7 !important;border-color:#002FA7 !important;color:white !important}
button:focus-visible {outline:2px solid #002FA7;outline-offset:2px}
button[kind="secondary"] {background:white !important;border-color:#ccc !important;color:#222 !important}
[data-testid="stAlert"] {background:#f6f7fa !important;color:#171717 !important;border:1px solid #e1e4ed !important}
[data-testid="stAlert"] p {color:#171717 !important}
[data-testid="stExpander"] {border-color:#e1e4ed !important}
[data-testid="stDataFrame"] {border:1px solid #e1e4ed}
[data-testid="stImage"] img {max-width:100%;border-radius:3px}
.app-header {border-bottom:1px solid #d9deea;padding-bottom:14px;margin-bottom:4px}
.app-name {font-size:1.8rem;font-weight:750;letter-spacing:-.045em}
.app-caption {font-size:.94rem;color:#475569;margin-top:3px}
.workflow-step {font-size:.9rem;color:#173d86;font-weight:700;margin:10px 0 0}
.rule-output {border:1px solid #cdd9f0;border-left:4px solid #002FA7;border-radius:8px;padding:16px 20px;
 background:#f2f6ff;font-size:1.2rem;font-weight:600;line-height:1.5}
@media(max-width:640px){.block-container{padding:18px 14px 30px}.app-name{font-size:1.4rem}}
</style>"""

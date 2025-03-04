from flask import Flask, jsonify
from datetime import datetime, timezone
import random

import requests

app = Flask(__name__)

BASE_URL_API = "https://api.nhle.com"
BASE_URL_API_WEB = "https://api-web.nhle.com"

global count

IS_GAME_LIVE = True
count = [-1]

@app.route("/v1/club-schedule/<club>/week/now")
def schedule(club):
    resp = requests.get(f"{BASE_URL_API_WEB}/v1/club-schedule/{club}/week/now").json()

    if IS_GAME_LIVE:
        print(datetime.now(timezone.utc))
        resp["games"][0]["startTimeUTC"] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

    return jsonify(resp)

@app.route("/stats/rest/en/team")
def teams():
    resp = requests.get(f"{BASE_URL_API}/stats/rest/en/team").json()

    return jsonify(resp)

@app.route("/v1/gamecenter/<match_id>/boxscore")
def game(match_id):
    resp = requests.get(f"{BASE_URL_API_WEB}/v1/gamecenter/{match_id}/boxscore").json()

    count[0] = count[0] + 1

    resp["gameState"] = "LIVE"
    resp["homeTeam"]["score"] = count[0]
    resp["awayTeam"]["score"] = count[0]

    print(resp["homeTeam"]["score"], resp["awayTeam"]["score"])

    return jsonify(resp)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
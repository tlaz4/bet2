from argparse import ArgumentParser
from dataclasses import dataclass
import math
import re

import cv2
import pytesseract
import numpy as np

@dataclass
class Probability:
    home_name: str
    home_prob: float
    away_name: str
    away_prob: float

def parse_viz_img(path):
    img_cv = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    img_str = pytesseract.image_to_string(img_rgb)
    print(img_str)

    home = re.findall(r'([0-9][0-9])%\n*([A-Z]+)', img_str)
    away = re.findall(r'([A-Z]+)\n*([0-9][0-9])%', img_str)

    return Probability(
            home_name=home[0][1], 
            home_prob=float(home[0][0]) / 100,
            away_name=away[0][0],
            away_prob=float(away[0][1]) / 100
    
    )

def parse_bet365_img(path):
    img_cv = cv2.imread(path)

    get_boxes_per_matchup(img_cv)

    img_cv = ~img_cv

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    img_str = pytesseract.image_to_string(img_rgb)

    print(re.findall(r'([A-Z]+.*[A-Za-z]+)', img_str))
    away_team, home_team = re.findall(r'([A-Z]+.*[A-Za-z]+)', img_str)
    away_prob, home_prob = re.findall(r'\n([0-9].[0-9]+)', img_str)

    return Probability(
        home_name=home_team,
        away_name=away_team,
        away_prob=float(away_prob),
        home_prob=float(home_prob)
    )

def get_boxes_per_matchup(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    height, width, channels = img_cv.shape
    green = (0, 255, 0)

    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=10,  # Max allowed gap between line for joining them
    )

    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]

        try:
            dist = math.sqrt((x2**2 - x1**2) + (y2**2 - y1**2))
        except ValueError:
            continue

        if dist >= width - 50:
            # Draw the lines joining the points
            # On the original image
            cv2.line(img_cv, (x1, y1), (x2, y2), green, 2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])

    cv2.line(img_cv, (0, 0), (width, 0), green, 2)
    lines_list.append([(0, 0), (width, 0)])

    print(len(lines_list))
    cv2.imwrite('./output.png', img_cv)


def parse_data_from_img(file_name):
    viz_prob = parse_viz_img(f"./viz/{file_name}")
    bet_365_prob = parse_bet365_img(f"./bet365/{file_name}")

    print(viz_prob)
    print(bet_365_prob)

    main(bet_365_prob.home_prob, bet_365_prob.away_prob, viz_prob.home_prob, viz_prob.away_prob, 100)

def odds_to_impl(odds1, odds2):
    return (1 / odds1 * 100, 1 / odds2 * 100)

def ev(prob, odds):
    amt_to_win = odds - 1
    return (prob * amt_to_win) - (1 - prob)

def kelly_criterion(impl_prob, odds, frac=0.4):
    return ((((odds - 1) *  impl_prob) - (1 - impl_prob)) / (odds - 1)) * frac

def main(home_odds, away_odds, home_prob, away_prob, bank_roll):
    impl_odds = odds_to_impl(home_odds, away_odds)
    kelly_fraction_home = kelly_criterion(home_prob, home_odds)
    kelly_fraction_away = kelly_criterion(away_prob, away_odds)

    print(f"Home Odds Probability: {impl_odds[0]}")
    print(f"Away Odds Probability: {impl_odds[1]}")
    print(f"Home Probability: {home_prob}")
    print(f"Away Probability: {away_prob}")
    print(f"Home Kelly Fraction: {kelly_fraction_home}")
    print(f"Away Kelly Fraction: {kelly_fraction_away}")
    
    if kelly_fraction_home > kelly_fraction_away:
        print(f"EV: {ev(home_prob, home_odds)}")
        print(f"Bet On Home: {kelly_fraction_home * bank_roll}")
    else:
        print(f"EV: {ev(away_prob, away_odds)}")
        print(f"Bet On Away: {kelly_fraction_away * bank_roll}")

if __name__ == "__main__":
    parser = ArgumentParser(prog='Odds Calculator', description='calculate odds')

    parser.add_argument('--home_odds', type=float)
    parser.add_argument('--away_odds', type=float)
    parser.add_argument('--home_prob', type=float)
    parser.add_argument('--away_prob', type=float)
    parser.add_argument('--bank_roll', type=int)

    args = parser.parse_args()

    main(args.home_odds, args.away_odds, args.home_prob, args.away_prob, args.bank_roll)


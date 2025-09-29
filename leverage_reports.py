import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os


TABLE_BLUE = colors.Color(65/255, 110/255, 220/255)
TABLE_GRAY = colors.Color(240/255, 240/255, 240/255)


class FooterCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_footer()
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self):
        width, height = self._pagesize
        self.setFillColor(TABLE_BLUE)
        self.rect(0, 0, width, 0.5 * inch, fill=1, stroke=0)
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.white)
        self.drawCentredString(width / 2, 0.2 * inch,
                               f"Page {self._pageNumber}")
        self.drawRightString(width - 0.5 * inch, 0.2 * inch,
                             "Stats via NCAA | d3-dashboard.com")


def format_baseball_ip(ip_value):
    if pd.isna(ip_value):
        return '-'

    if isinstance(ip_value, str):
        try:
            ip_value = float(ip_value)
        except ValueError:
            return '-'

    full_innings = int(ip_value)
    partial = ip_value - full_innings

    if partial < 0.17:
        outs = 0
    elif partial < 0.50:
        outs = 1
    elif partial < 0.84:
        outs = 2
    else:
        full_innings += 1
        outs = 0

    return f"{full_innings}.{outs}"


def calculate_leverage_stats(pbp_combined, team_filter=None, days_back=7):
    today = datetime.now()
    last_week = today - timedelta(days=days_back)

    pbp_combined['date'] = pd.to_datetime(pbp_combined['date'])
    recent_pbp = pbp_combined[pbp_combined['date'] >= last_week]

    if team_filter:
        recent_pbp = recent_pbp[recent_pbp['pitch_team'].isin(team_filter)]

    recent_pbp_sorted = recent_pbp.sort_values(
        ['game_id', 'inning', 'play_id'])

    first_li_per_game = (recent_pbp_sorted
                         .groupby(['pitcher_standardized', 'game_id'])
                         .apply(lambda x: pd.Series({
                             'li': x['li'].iloc[0] if x['inning'].iloc[0] != 1 else np.nan,
                             'inning': x['inning'].iloc[0],
                             'pitch_team': x['pitch_team'].iloc[0]
                         }))
                         .reset_index())

    first_li_per_inning = (recent_pbp_sorted
                           .groupby(['pitcher_standardized', 'game_id', 'inning'])
                           .apply(lambda x: pd.Series({
                               'li': x['li'].iloc[0],
                               'pitch_team': x['pitch_team'].iloc[0]
                           }))
                           .reset_index())

    last_li_per_game = (recent_pbp_sorted
                        .groupby(['pitcher_standardized', 'game_id'])
                        .apply(lambda x: pd.Series({
                            'li': x['li'].iloc[-1],
                            'inning': x['inning'].iloc[-1],
                            'pitch_team': x['pitch_team'].iloc[-1]
                        }))
                        .reset_index())

    gmli_by_pitcher = (first_li_per_game
                       .groupby(['pitcher_standardized'])
                       .agg({'li': 'mean'})
                       .reset_index()
                       .rename(columns={'li': 'gmLI'})
                       .set_index('pitcher_standardized')
                       .to_dict()['gmLI'])

    inli_by_pitcher = (first_li_per_inning
                       .groupby(['pitcher_standardized'])
                       .agg({'li': 'mean'})
                       .reset_index()
                       .rename(columns={'li': 'inLI'})
                       .set_index('pitcher_standardized')
                       .to_dict()['inLI'])

    exli_by_pitcher = (last_li_per_game
                       .groupby(['pitcher_standardized'])
                       .agg({'li': 'mean'})
                       .reset_index()
                       .rename(columns={'li': 'exLI'})
                       .set_index('pitcher_standardized')
                       .to_dict()['exLI'])

    pitcher_stats = {}

    outs = [2, 3]
    batter_events = [2, 3, 14, 15, 16, 18, 19, 20, 21, 22, 23]
    single = 20
    double = 21
    triple = 22
    hr = 23
    walks = [14, 15]
    hbp = 16
    so = 3

    woba_weights = {
        'BB': 0.8,
        'HBP': 0.8,
        '1B': 0.9,
        '2B': 1.3,
        '3B': 1.6,
        'HR': 1.8
    }

    pitcher_game_groups = recent_pbp.groupby(
        ['pitcher_standardized', 'pitch_team', 'game_id'])

    for (pitcher, team, game_id), game_data in pitcher_game_groups:
        if pd.isna(pitcher) or pitcher == '':
            continue

        if pitcher not in pitcher_stats:
            pitcher_stats[pitcher] = {
                'team': team,
                'games': set(),
                'batters_faced': 0,
                'ip': 0,
                'k': 0,
                'wpa': 0,
                're24': 0,
                'bb': 0,
                'hbp': 0,
                'h': 0,
                '1b_allowed': 0,
                '2b_allowed': 0,
                '3b_allowed': 0,
                'hr_allowed': 0,
                'ab_faced': 0,
                'sf_faced': 0,
                'hits_allowed': 0
            }

        pitcher_stats[pitcher]['games'].add(game_id)

        outs_recorded = game_data.outs_on_play.sum()
        pitcher_stats[pitcher]['ip'] += outs_recorded / 3
        pitcher_stats[pitcher]['k'] += len(
            game_data[game_data['event_cd'] == so])
        pitcher_stats[pitcher]['bb'] += len(
            game_data[game_data['event_cd'].isin(walks)])
        pitcher_stats[pitcher]['hbp'] += len(
            game_data[game_data['event_cd'] == hbp])

        batters_faced = len(
            game_data[game_data['event_cd'].isin(batter_events)])
        pitcher_stats[pitcher]['batters_faced'] += batters_faced

        pitcher_stats[pitcher]['wpa'] -= game_data['wpa'].sum(
        ) if 'wpa' in game_data.columns else 0
        pitcher_stats[pitcher]['re24'] -= game_data['rea'].sum(
        ) if 'rea' in game_data.columns else 0

        pitcher_stats[pitcher]['1b_allowed'] += len(
            game_data[game_data['event_cd'] == single])
        pitcher_stats[pitcher]['2b_allowed'] += len(
            game_data[game_data['event_cd'] == double])
        pitcher_stats[pitcher]['3b_allowed'] += len(
            game_data[game_data['event_cd'] == triple])
        pitcher_stats[pitcher]['hr_allowed'] += len(
            game_data[game_data['event_cd'] == hr])

        hits = len(game_data[game_data['event_cd'].isin(
            [single, double, triple, hr])])
        pitcher_stats[pitcher]['hits_allowed'] += hits

        pitcher_stats[pitcher]['ab_faced'] += len(
            game_data[game_data['event_cd'].isin(batter_events)])
        pitcher_stats[pitcher]['sf_faced'] += len(
            game_data[game_data['sf_fl'] == 1]) if 'sf_fl' in game_data.columns else 0

    results = []
    for pitcher, stats in pitcher_stats.items():
        gmli = gmli_by_pitcher.get(pitcher, 0)
        exli = exli_by_pitcher.get(pitcher, 0)
        inli = inli_by_pitcher.get(pitcher, 0)

        woba_numerator = (
            woba_weights['BB'] * stats['bb'] +
            woba_weights['HBP'] * stats['hbp'] +
            woba_weights['1B'] * stats['1b_allowed'] +
            woba_weights['2B'] * stats['2b_allowed'] +
            woba_weights['3B'] * stats['3b_allowed'] +
            woba_weights['HR'] * stats['hr_allowed']
        )

        woba_denominator = stats['ab_faced'] + \
            stats['bb'] + stats['sf_faced'] + stats['hbp']
        o_woba = woba_numerator / woba_denominator if woba_denominator > 0 else 0

        opp_ba = stats['hits_allowed'] / \
            stats['ab_faced'] if stats['ab_faced'] > 0 else 0

        on_base_events = stats['hits_allowed'] + stats['bb'] + stats['hbp']
        plate_appearances = stats['ab_faced'] + \
            stats['bb'] + stats['hbp'] + stats['sf_faced']
        opp_obp = on_base_events / plate_appearances if plate_appearances > 0 else 0

        total_bases = (
            stats['1b_allowed'] * 1 +
            stats['2b_allowed'] * 2 +
            stats['3b_allowed'] * 3 +
            stats['hr_allowed'] * 4
        )
        opp_slg = total_bases / \
            stats['ab_faced'] if stats['ab_faced'] > 0 else 0

        results.append({
            'Pitcher': pitcher,
            'Team': stats['team'],
            'G': len(stats['games']),
            'IP': stats['ip'],
            'BF': stats['batters_faced'],
            'H': int(stats['hits_allowed']),
            'K': int(stats['k']),
            'BB': int(stats['bb']),
            'HBP': int(stats['hbp']),
            'o-wOBA': round(o_woba, 3),
            'gmLI': round(gmli, 2),
            'exLI': round(exli, 2),
            'inLI': round(inli, 2),
            'WPA': round(stats['wpa'], 2),
            'RE24': round(stats['re24'], 2)
        })

    results_df = pd.DataFrame(results).sort_values('gmLI', ascending=False)

    return results_df


def create_leverage_report_pdf(stats_df, output_filename, start_date_str=None):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=landscape(letter),
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=20,
        fontName='Helvetica-Bold',
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica',
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=15
    )

    if start_date_str:
        subtitle_text = start_date_str
    else:
        today = datetime.now()
        last_week = today - timedelta(days=7)
        subtitle_text = f"Week 1 | {last_week.strftime('%b. %d')}-{today.strftime('%d')}"

    team_groups = stats_df.groupby('Team')

    col_widths = [
        1.8*inch,  # Pitcher
        0.55*inch,  # G
        0.55*inch,  # IP
        0.55*inch,  # BF
        0.55*inch,  # H
        0.55*inch,  # K
        0.55*inch,  # BB
        0.55*inch,  # HBP
        0.8*inch,  # o-wOBA
        0.6*inch,  # gmLI
        0.6*inch,  # exLI
        0.6*inch,  # inLI
        0.6*inch,  # WPA
        0.7*inch   # RE24
    ]

    for team_name, team_df in team_groups:
        title = Paragraph(f"Leverage Index Report - {team_name}", title_style)
        subtitle = Paragraph(subtitle_text, subtitle_style)
        elements.append(title)
        elements.append(subtitle)

        header_data = [['Pitcher', 'G', 'IP', 'BF', 'H', 'K', 'BB',
                        'HBP', 'o-wOBA', 'gmLI', 'exLI', 'inLI', 'WPA', 'RE24']]
        header_table = Table(header_data, colWidths=col_widths)
        header_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGNMENT', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, 0), 1, colors.white),
        ])
        header_table.setStyle(header_style)
        elements.append(header_table)

        data = []

        for _, row in team_df.iterrows():
            data.append([
                row['Pitcher'],
                str(int(row['G'])),
                format_baseball_ip(row['IP']),
                str(int(row['BF'])),
                str(int(row['H'])),
                str(int(row['K'])),
                str(int(row['BB'])),
                str(int(row['HBP'])),
                '-' if pd.isna(row['o-wOBA']) else str(row['o-wOBA']),
                '-' if pd.isna(row['gmLI']) else str(row['gmLI']),
                '-' if pd.isna(row['exLI']) else str(row['exLI']),
                format_baseball_ip(row['inLI']),
                '-' if pd.isna(row['WPA']) else str(row['WPA']),
                '-' if pd.isna(row['RE24']) else str(row['RE24'])
            ])

        data_table = Table(data, colWidths=col_widths)

        data_style = TableStyle([
            ('ALIGNMENT', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGNMENT', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.lightgrey),
            ('LINEAFTER', (0, 0), (-2, -1), 0.5, colors.lightgrey),
        ])

        for i in range(0, len(data), 2):
            data_style.add('BACKGROUND', (0, i), (-1, i), TABLE_GRAY)

        data_table.setStyle(data_style)
        elements.append(data_table)
        elements.append(Spacer(1, 15))
        elements.append(PageBreak())

    doc.build(elements, canvasmaker=FooterCanvas)
    print(f"PDF created: {output_filename}")


def generate_leverage_report(pbp, teams=None, output_filename="leverage_report.pdf"):
    stats_df = calculate_leverage_stats(pbp, team_filter=teams)

    if stats_df.empty:
        print("No data found for the specified teams or time period.")
        return

    create_leverage_report_pdf(stats_df, output_filename)


def send_email_with_pdf(pdf_path, recipient_emails):
    sender_email = "jackkelly12902@gmail.com"
    sender_password = "pxvu hjsp sgjz wvlm"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipient_emails)
    msg['Subject'] = f"Daily Longball Labs Report - {datetime.now().strftime('%Y-%m-%d')}"

    body = "Attached are automated leverage reports for all NTangible team clients. Please contact Jack Kelly with any questions or concerns."
    msg.attach(MIMEText(body, 'plain'))

    with open(pdf_path, "rb") as f:
        attach = MIMEApplication(f.read(), _subtype="pdf")
    attach.add_header('Content-Disposition', 'attachment',
                      filename=os.path.basename(pdf_path))
    msg.attach(attach)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent successfully to {', '.join(recipient_emails)}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def main(data_dir, year, recipient_emails):
    data_dir = Path(data_dir)
    nt_teams = ['Hofstra', 'Michigan St.', 'Northwood', 'UIndy', 'Boston College',
                'Virginia', 'Baylor', 'Texas A&M', 'Arizona St.', 'Indiana', 'Clemson', 'South Alabama']

    pbp_d1 = pd.read_csv(data_dir /
                         f'play_by_play/d1_parsed_pbp_new_{year}.csv')
    pbp_d2 = pd.read_csv(data_dir /
                         f'play_by_play/d2_parsed_pbp_new_{year}.csv')
    pbp_d3 = pd.read_csv(data_dir /
                         f'play_by_play/d3_parsed_pbp_new_{year}.csv')

    pbp_combined = pd.concat([pbp_d1, pbp_d2, pbp_d3], ignore_index=True)

    generate_leverage_report(pbp_combined, nt_teams,
                             output_filename="leverage_report.pdf")
    send_email_with_pdf('leverage_report.pdf', recipient_emails)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    parser.add_argument('--year', required=True)
    parser.add_argument('--recipient_emails', required=True)

    args = parser.parse_args()

    main(args.data_dir, args.year, args.recipient_emails)

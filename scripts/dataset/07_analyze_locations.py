"""
Script 7: Analisi geografica dei samples 2021/2022
Reverse geocoding per identificare i paesi
"""

import pandas as pd
from collections import Counter

def main():
    print("=" * 60)
    print("ANALISI GEOGRAFICA SAMPLES 2021/2022")
    print("=" * 60)

    # Prova a importare reverse_geocoder in modalità single-thread
    try:
        import reverse_geocoder as rg
        USE_RG = True
        print("Usando reverse_geocoder (modalita single-thread)")
    except ImportError:
        USE_RG = False
        print("reverse_geocoder non installato, uso fallback")

    # Bounding box approssimati dei paesi (fallback)
    COUNTRY_BOXES = {
        'Australia': {'lon': (113, 154), 'lat': (-44, -10)},
        'New Zealand': {'lon': (166, 179), 'lat': (-47, -34)},
        'USA (Continental)': {'lon': (-125, -66), 'lat': (24, 49)},
        'Canada': {'lon': (-141, -52), 'lat': (49, 84)},
        'Alaska (USA)': {'lon': (-180, -130), 'lat': (51, 72)},
        'Russia': {'lon': (19, 180), 'lat': (50, 82)},
        'China': {'lon': (73, 135), 'lat': (18, 54)},
        'Brazil': {'lon': (-74, -34), 'lat': (-34, 5)},
        'Argentina': {'lon': (-74, -53), 'lat': (-56, -21)},
        'Chile': {'lon': (-76, -66), 'lat': (-56, -17)},
        'Bolivia': {'lon': (-70, -57), 'lat': (-23, -9)},
        'South Africa': {'lon': (16, 33), 'lat': (-35, -22)},
        'Namibia': {'lon': (11, 25), 'lat': (-29, -17)},
        'Angola': {'lon': (11, 24), 'lat': (-18, -4)},
        'Spain': {'lon': (-10, 5), 'lat': (35, 44)},
        'Portugal': {'lon': (-10, -6), 'lat': (36, 42)},
        'Italy': {'lon': (6, 19), 'lat': (36, 47)},
        'Austria': {'lon': (9, 17), 'lat': (46, 49)},
        'Germany': {'lon': (5, 16), 'lat': (47, 55)},
        'France': {'lon': (-5, 10), 'lat': (41, 51)},
        'Norway': {'lon': (4, 31), 'lat': (57, 72)},
        'Sweden': {'lon': (11, 24), 'lat': (55, 69)},
        'Finland': {'lon': (19, 32), 'lat': (59, 70)},
        'Kazakhstan': {'lon': (46, 88), 'lat': (40, 56)},
        'Mongolia': {'lon': (87, 120), 'lat': (41, 52)},
        'Japan': {'lon': (129, 146), 'lat': (30, 46)},
        'Indonesia': {'lon': (95, 141), 'lat': (-11, 6)},
        'Philippines': {'lon': (116, 127), 'lat': (4, 21)},
        'India': {'lon': (68, 98), 'lat': (6, 36)},
        'Tibet/Nepal': {'lon': (80, 92), 'lat': (26, 36)},
        'Saudi Arabia': {'lon': (34, 56), 'lat': (16, 33)},
        'Sudan': {'lon': (21, 39), 'lat': (8, 23)},
        'Ethiopia': {'lon': (33, 48), 'lat': (3, 15)},
        'Algeria': {'lon': (-9, 12), 'lat': (18, 38)},
        'Greenland': {'lon': (-73, -11), 'lat': (59, 84)},
        'Iceland': {'lon': (-25, -13), 'lat': (63, 67)},
        'Antarctica': {'lon': (-180, 180), 'lat': (-90, -60)},
        'Arctic (>75N)': {'lon': (-180, 180), 'lat': (75, 90)},
    }

    # Mappa codici paese ISO a nomi
    COUNTRY_NAMES = {
        'US': 'USA', 'CA': 'Canada', 'RU': 'Russia', 'CN': 'China',
        'AU': 'Australia', 'NZ': 'New Zealand', 'BR': 'Brazil',
        'AR': 'Argentina', 'CL': 'Chile', 'BO': 'Bolivia',
        'ZA': 'South Africa', 'NA': 'Namibia', 'AO': 'Angola',
        'ES': 'Spain', 'PT': 'Portugal', 'IT': 'Italy', 'AT': 'Austria',
        'DE': 'Germany', 'FR': 'France', 'NO': 'Norway', 'SE': 'Sweden',
        'FI': 'Finland', 'KZ': 'Kazakhstan', 'MN': 'Mongolia',
        'JP': 'Japan', 'ID': 'Indonesia', 'PH': 'Philippines',
        'IN': 'India', 'NP': 'Nepal', 'SA': 'Saudi Arabia',
        'SD': 'Sudan', 'ET': 'Ethiopia', 'DZ': 'Algeria',
        'GL': 'Greenland', 'IS': 'Iceland', 'AQ': 'Antarctica',
        'MX': 'Mexico', 'CO': 'Colombia', 'PE': 'Peru', 'VE': 'Venezuela',
        'EC': 'Ecuador', 'UY': 'Uruguay', 'PY': 'Paraguay',
        'EG': 'Egypt', 'LY': 'Libya', 'MA': 'Morocco', 'TN': 'Tunisia',
        'KE': 'Kenya', 'TZ': 'Tanzania', 'UG': 'Uganda', 'NG': 'Nigeria',
        'GH': 'Ghana', 'SN': 'Senegal', 'ML': 'Mali', 'NE': 'Niger',
        'TD': 'Chad', 'CF': 'Central African Rep.', 'CD': 'DR Congo',
        'ZW': 'Zimbabwe', 'BW': 'Botswana', 'MZ': 'Mozambique',
        'GB': 'UK', 'IE': 'Ireland', 'NL': 'Netherlands', 'BE': 'Belgium',
        'CH': 'Switzerland', 'PL': 'Poland', 'CZ': 'Czech Republic',
        'SK': 'Slovakia', 'HU': 'Hungary', 'RO': 'Romania', 'BG': 'Bulgaria',
        'GR': 'Greece', 'TR': 'Turkey', 'UA': 'Ukraine', 'BY': 'Belarus',
        'LT': 'Lithuania', 'LV': 'Latvia', 'EE': 'Estonia',
        'IR': 'Iran', 'IQ': 'Iraq', 'SY': 'Syria', 'JO': 'Jordan',
        'IL': 'Israel', 'LB': 'Lebanon', 'AE': 'UAE', 'OM': 'Oman',
        'YE': 'Yemen', 'PK': 'Pakistan', 'AF': 'Afghanistan',
        'UZ': 'Uzbekistan', 'TM': 'Turkmenistan', 'TJ': 'Tajikistan',
        'KG': 'Kyrgyzstan', 'BD': 'Bangladesh', 'MM': 'Myanmar',
        'TH': 'Thailand', 'VN': 'Vietnam', 'LA': 'Laos', 'KH': 'Cambodia',
        'MY': 'Malaysia', 'SG': 'Singapore', 'KR': 'South Korea',
        'KP': 'North Korea', 'TW': 'Taiwan', 'PG': 'Papua New Guinea',
        'FJ': 'Fiji', 'SJ': 'Svalbard', 'AX': 'Aland Islands',
    }

    def get_country_fallback(lon, lat):
        """Identifica paese usando bounding box (fallback)"""
        for country, box in COUNTRY_BOXES.items():
            lon_min, lon_max = box['lon']
            lat_min, lat_max = box['lat']
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                return country
        return f"Unknown ({lon:.1f}, {lat:.1f})"

    def analyze_year(year):
        """Analizza samples di un anno"""
        csv_path = f"samples_{year}_global.csv"

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  File {csv_path} non trovato")
            return None

        print(f"\n  Totale samples: {len(df)}")

        countries = []
        cities = []

        for idx, row in df.iterrows():
            lat, lon = row['lat'], row['lon']

            if USE_RG:
                # Single query (evita multiprocessing)
                result = rg.search([(lat, lon)], mode=1)[0]
                cc = result['cc']
                country = COUNTRY_NAMES.get(cc, cc)
                city = result['name']
            else:
                country = get_country_fallback(lon, lat)
                city = ''

            countries.append(country)
            cities.append(city)

        df['country'] = countries
        df['city'] = cities

        return df

    # Analizza entrambi gli anni
    all_data = {}

    for year in [2021, 2022]:
        print(f"\n{'='*60}")
        print(f"ANNO {year}")
        print("="*60)

        df = analyze_year(year)
        if df is not None:
            all_data[year] = df

            # Conta per paese
            country_counts = Counter(df['country'])

            print(f"\n  DISTRIBUZIONE PER PAESE:")
            print(f"  {'-'*40}")
            for country, count in country_counts.most_common():
                pct = 100 * count / len(df)
                print(f"  {country:<25} {count:>3} ({pct:>5.1f}%)")

    # Riepilogo totale
    print(f"\n{'='*60}")
    print("RIEPILOGO TOTALE")
    print("="*60)

    if all_data:
        combined = pd.concat(all_data.values(), ignore_index=True)
        total_counts = Counter(combined['country'])

        print(f"\nTotale samples: {len(combined)}")
        print(f"\nDISTRIBUZIONE GLOBALE:")
        print(f"{'-'*50}")
        for country, count in total_counts.most_common():
            pct = 100 * count / len(combined)
            bar = '#' * int(pct / 2)
            print(f"{country:<25} {count:>3} ({pct:>5.1f}%) {bar}")

        # Salva dettagli
        for year, df in all_data.items():
            out_path = f"samples_{year}_with_location.csv"
            df.to_csv(out_path, index=False)
            print(f"\nSalvato: {out_path}")

if __name__ == '__main__':
    main()

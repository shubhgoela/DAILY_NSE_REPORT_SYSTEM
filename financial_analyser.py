import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np

def df_to_styled_html(df: pd.DataFrame) -> str:
    def format_cell(col, val):
        style = ''
        formatted_val = val

        if col in ['pct_ChngInOptnPric', 'pct_ChngInOpnIntrst', 'pct_ChngInFutPric']:
            try:
                num = float(val)
                style = 'background-color:rgb(106,168,79);' if num > 0 else 'background-color:rgb(224,102,102);' if num < 0 else ''
                formatted_val = f'{num:.2f}%'
            except:
                style = 'background-color:yellow;'
                formatted_val = val if pd.notna(val) else 'N/A'

        elif col in ['ChngInOpnIntrst']:
            try:
                num = float(val)
                style = 'background-color:rgb(106,168,79);' if num > 0 else 'background-color:rgb(224,102,102);' if num < 0 else ''
                formatted_val = f'{num:,.2f}'
            except:
                style = 'background-color:yellow;'
                formatted_val = val if pd.notna(val) else 'N/A'

        elif col in ['OpnIntrst', 'prev_OpnIntrst', 'SttlmPric', 'PrvsClsgPric', 'StrkPric']:
            try:
                num = float(val)
                formatted_val = f'{num:,.2f}'
            except:
                formatted_val = val

        style += 'text-align:center;'  # Center align all cells
        return formatted_val, style

    html = '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;font-family:Arial;font-size:12px;">'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th style="background-color:#f2f2f2;">{col}</th>'
    html += '</tr></thead><tbody>'

    for _, row in df.iterrows():
        row_style = 'background-color:yellow;' if row.get('OptnTp') == 'CMP' else ''
        html += f'<tr style="{row_style}">'
        for col in df.columns:
            val = row[col]
            formatted_val, cell_style = format_cell(col, val)
            html += f'<td style="{cell_style}">{formatted_val}</td>'
        html += '</tr>'
    html += '</tbody></table>'

    return html



class FinancialDataAnalyzer:
    def __init__(self, fo_bhav_copy, symbol, month_expry, curr_close, prev_close):
        """
        Initialize with actual trading data
        
        Args:
            fo_bhav_copy: DataFrame containing futures and options data
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY')
            month_expry: Expiry date for current month
            curr_close: Current closing price
            prev_close: Previous closing price
        """
        self.fo_bhav_copy = fo_bhav_copy
        self.symbol = symbol
        self.month_expry = month_expry
        self.curr_close = curr_close
        self.prev_close = prev_close
        
        # Generate tables using provided functions
        self.fut_oi_table = self.create_futures_data_table()
        self.oi_table = self.create_OI_table()
        self.curr_atm_oi = self.get_atm_and_oi_from_price(curr_close)
        self.prev_atm_oi = self.get_atm_and_oi_from_price(prev_close)
        self.total_put_oi, self.total_call_oi, self.pcr, self.total_oi = self.get_pcr_and_total_oi()

        self.analysis_results = {}
    
    def create_futures_data_table(self, columns=['XpryDt', 'SttlmPric','PrvsClsgPric','pct_ChngInFutPric', 
                                                'OpnIntrst', 'prev_OpnIntrst', 'ChngInOpnIntrst','pct_ChngInOpnIntrst']):
        """Create futures data table using the provided function logic"""
        df = self.fo_bhav_copy[(self.fo_bhav_copy['TckrSymb'] == self.symbol) & 
                    (self.fo_bhav_copy['FinInstrmTp'] == 'STF')].copy()

        if df.empty:
            return pd.DataFrame()

        df['prev_OpnIntrst'] = df['OpnIntrst'] - df['ChngInOpnIntrst']
        df['pct_ChngInFutPric'] = np.where(df['PrvsClsgPric'] != 0,
                                         round((df['SttlmPric'] - df['PrvsClsgPric']) / df['PrvsClsgPric'] * 100, 2), 0)
        numeric_cols = ['OpnIntrst', 'prev_OpnIntrst', 'ChngInOpnIntrst']

        # Create total row
        total_row = {col: df[col].sum() if col in numeric_cols else '' for col in df.columns}
        total_row[self.symbol] = 'Total'
        total_row['XpryDt'] = 'Total'

        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        df['pct_ChngInOpnIntrst'] = np.where(df['prev_OpnIntrst'] != 0, 
                                           round((df['ChngInOpnIntrst'] / df['prev_OpnIntrst']) * 100, 2), 0)
        
        return df[columns].reset_index(drop=True)

    def create_OI_table(self, columns_to_return=['StrkPric', 'OptnTp','SttlmPric','PrvsClsgPric',
                                               'pct_ChngInOptnPric', 'OpnIntrst', 'prev_OpnIntrst', 
                                               'ChngInOpnIntrst','pct_ChngInOpnIntrst']):
        """Create OI table using the provided function logic"""
        XpryDts = [self.month_expry]
        
        row_labels = [
            ('Max Call OI', 'CE', 'OpnIntrst', 'idxmax'),
            ('Max Put OI', 'PE', 'OpnIntrst', 'idxmax'),
            ('Min Call OI', 'CE', 'OpnIntrst', 'idxmin'),
            ('Min Put OI', 'PE', 'OpnIntrst', 'idxmin'),
            ('Max Call OI addition', 'CE', 'ChngInOpnIntrst', 'idxmax'),
            ('Max Put OI addition', 'PE', 'ChngInOpnIntrst', 'idxmax'),
            ('Max Call OI unwinding', 'CE', 'ChngInOpnIntrst', 'idxmin'),
            ('Max Put OI unwinding', 'PE', 'ChngInOpnIntrst', 'idxmin'),
        ]

        # Initialize dataframe
        df = pd.DataFrame({
            self.symbol: [label for label, _, _, _ in row_labels],
            'Option Type': [opt_type for _, opt_type, _, _ in row_labels]
        })

        # Process each expiry date
        for date in XpryDts:
            date_str = date.strftime('%Y-%m-%d')
            
            result_rows = []
            for label, opt_type, ref_col, func in row_labels:
                subset = self.fo_bhav_copy[
                    (self.fo_bhav_copy['XpryDt'] == date_str) &
                    (self.fo_bhav_copy['TckrSymb'] == self.symbol) &
                    (self.fo_bhav_copy['OptnTp'] == opt_type)
                ]

                if subset.empty or ref_col not in subset.columns:
                    result_rows.append([None] * len(columns_to_return))
                    continue

                try:
                    idx = getattr(subset[ref_col], func)()
                    selected_row = subset.loc[idx]
                    result_rows.append([selected_row.get(col, None) for col in columns_to_return])
                except Exception:
                    result_rows.append([None] * len(columns_to_return))

            # Add columns
            for i, col in enumerate(columns_to_return):
                df[col] = [row[i] if row else None for row in result_rows]
        
        # Add total row
        numeric_cols = ['SttlmPric', 'PrvsClsgPric', 'OpnIntrst', 'prev_OpnIntrst', 'ChngInOpnIntrst']
        
        total_row = {col: df[col].sum() if col in numeric_cols else '' for col in df.columns}
        total_row[self.symbol] = 'Total'

        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        # Calculate derived fields safely
        df['prev_OpnIntrst'] = df['OpnIntrst'] - df['ChngInOpnIntrst']

        df['pct_ChngInOpnIntrst'] = df.apply(
            lambda row: round((row['ChngInOpnIntrst'] / row['prev_OpnIntrst']) * 100, 2)
            if row['prev_OpnIntrst'] not in [0, None, np.nan]
            else 0,
            axis=1
        )

        df['pct_ChngInOptnPric'] = df.apply(
            lambda row: round((row['SttlmPric'] - row['PrvsClsgPric']) / row['PrvsClsgPric'] * 100, 2)
            if pd.notna(row['PrvsClsgPric']) and row['PrvsClsgPric'] != 0
            else 0,
            axis=1
        )

        df = df.fillna(0)
        return df

    def get_atm_and_oi_from_price(self, price, return_cols=['XpryDt', 'StrkPric', 'OptnTp','SttlmPric',
                                                          'PrvsClsgPric','pct_ChngInOptnPric', 'OpnIntrst', 
                                                          'prev_OpnIntrst', 'ChngInOpnIntrst','pct_ChngInOpnIntrst']):
        """Get ATM options data using the provided function logic"""
        XpryDt = self.month_expry
        
        # Filter data
        fo_bhav_copy_filtered = self.fo_bhav_copy[
            (self.fo_bhav_copy['TckrSymb'] == self.symbol) & 
            (self.fo_bhav_copy['FinInstrmTp'] == 'STO') & 
            (self.fo_bhav_copy['XpryDt'] == XpryDt.strftime('%Y-%m-%d'))
        ].copy()
        
        if fo_bhav_copy_filtered.empty:
            return pd.DataFrame()
        
        fo_bhav_copy_filtered['abs_from_strk'] = (fo_bhav_copy_filtered['StrkPric'] - price).abs()
        fo_bhav_copy_filtered = fo_bhav_copy_filtered.sort_values(by='abs_from_strk').head(4)
        
        # Add current price row
        current_price_row = {
            'TckrSymb': self.symbol,
            'XpryDt': None,
            'StrkPric': price,
            'OptnTp': 'CMP',
            'OpnIntrst': None,
            'prev_OpnIntrst': None,
            'ChngInOpnIntrst': None,
            'pct_ChngInOpnIntrst': None,
            'SttlmPric': None,
            'PrvsClsgPric': None,
            'pct_ChngInOptnPric': None,
            'abs_from_strk': 0
        }

        fo_bhav_copy_filtered = pd.concat([
            fo_bhav_copy_filtered,
            pd.DataFrame([current_price_row])
        ], ignore_index=True).sort_values(by='StrkPric')

        # Add total row
        numeric_cols = ['SttlmPric', 'PrvsClsgPric', 'OpnIntrst', 'prev_OpnIntrst', 'ChngInOpnIntrst']
        
        total_row = {col: fo_bhav_copy_filtered[col].sum() if col in numeric_cols else '' for col in fo_bhav_copy_filtered.columns}
        total_row[self.symbol] = 'Total'
        total_row['XpryDt'] = 'Total'

        fo_bhav_copy_filtered = pd.concat([fo_bhav_copy_filtered, pd.DataFrame([total_row])], ignore_index=True)

        # Calculate derived fields safely
        fo_bhav_copy_filtered['prev_OpnIntrst'] = fo_bhav_copy_filtered['OpnIntrst'] - fo_bhav_copy_filtered['ChngInOpnIntrst']
        fo_bhav_copy_filtered['pct_ChngInOpnIntrst'] = np.where(fo_bhav_copy_filtered['prev_OpnIntrst'] != 0,
                                                              round((fo_bhav_copy_filtered['ChngInOpnIntrst'] / fo_bhav_copy_filtered['prev_OpnIntrst']) * 100, 2), 0)
        fo_bhav_copy_filtered['pct_ChngInOptnPric'] = np.where(fo_bhav_copy_filtered['PrvsClsgPric'] != 0,
                                                             round((fo_bhav_copy_filtered['SttlmPric'] - fo_bhav_copy_filtered['PrvsClsgPric']) / fo_bhav_copy_filtered['PrvsClsgPric'] * 100, 2), 0)
        
        fo_bhav_copy_filtered = fo_bhav_copy_filtered.fillna('')
        fo_bhav_copy_filtered.reset_index(drop=True, inplace=True)

        return fo_bhav_copy_filtered[return_cols] if return_cols else fo_bhav_copy_filtered

    def get_pcr_and_total_oi(self, ref_date=None):
        """
        Calculate the Put-Call Ratio (PCR) for a given index and expiry date.
        """
        XpryDt = self.month_expry
        index = self.symbol

        if ref_date is None:
            ref_date = datetime.today()
        elif isinstance(ref_date, str):
            ref_date = datetime.strptime(ref_date, "%Y-%m-%d")

        if ref_date.date() == XpryDt:
            fo_data_filterd = self.fo_bhav_copy[(self.fo_bhav_copy['XpryDt'] != ref_date.strftime('%Y-%m-%d')) & (self.fo_bhav_copy['TckrSymb'] == index)]
        else:
            fo_data_filterd = self.fo_bhav_copy[(self.fo_bhav_copy['TckrSymb'] == index)]

        fo_data_filterd_ce = fo_data_filterd[(fo_data_filterd['OptnTp'] == 'CE')]
        fo_data_filterd_pe = fo_data_filterd[(fo_data_filterd['OptnTp'] == 'PE')]

        put_oi = fo_data_filterd_pe['OpnIntrst'].sum()
        call_oi = fo_data_filterd_ce['OpnIntrst'].sum()

        pcr = round((put_oi/call_oi), 2) if call_oi != 0 else 0
        total_oi = put_oi + call_oi
        
        return put_oi, call_oi, pcr, total_oi

    def analyze_futures_market(self):
        """Analyze futures market data"""
        if self.fut_oi_table.empty:
            return {}
            
        # Get total row (last row)
        total_row = self.fut_oi_table.iloc[-1]
        
        analysis = {
            'total_oi': self._safe_float(total_row.get('OpnIntrst', 0)),
            'oi_change': self._safe_float(total_row.get('ChngInOpnIntrst', 0)),
            'oi_change_pct': self._safe_float(total_row.get('pct_ChngInOpnIntrst', 0)),
            'near_month_data': {}
        }
        
        # Get near month data (first row, excluding total)
        if len(self.fut_oi_table) > 1:
            near_month = self.fut_oi_table.iloc[0]
            analysis['near_month_data'] = {
                'expiry': near_month.get('XpryDt', ''),
                'settlement_price': self._safe_float(near_month.get('SttlmPric', 0)),
                'prev_close': self._safe_float(near_month.get('PrvsClsgPric', 0)),
                'price_change_pct': self._safe_float(near_month.get('pct_ChngInFutPric', 0)),
                'oi': self._safe_float(near_month.get('OpnIntrst', 0)),
                'oi_change': self._safe_float(near_month.get('ChngInOpnIntrst', 0)),
                'oi_change_pct': self._safe_float(near_month.get('pct_ChngInOpnIntrst', 0))
            }
        
        return analysis

    def analyze_options_activity(self):
        """Analyze current ATM options activity"""
        if self.curr_atm_oi.empty:
            return {}
            
        # Find current market price (CMP row)
        cmp_rows = self.curr_atm_oi[self.curr_atm_oi['OptnTp'] == 'CMP']
        current_price = self.curr_close
        if not cmp_rows.empty:
            current_price = self._safe_float(cmp_rows.iloc[0]['StrkPric'])
        
        # Analyze strike-wise data
        strikes_analysis = {}
        
        for _, row in self.curr_atm_oi.iterrows():
            if row['OptnTp'] in ['CE', 'PE']:
                strike = self._safe_float(row['StrkPric'])
                option_type = row['OptnTp']
                
                if strike not in strikes_analysis:
                    strikes_analysis[strike] = {}
                
                strikes_analysis[strike][option_type] = {
                    'settlement_price': self._safe_float(row['SttlmPric']),
                    'price_change_pct': self._safe_float(row['pct_ChngInOptnPric']),
                    'oi': self._safe_float(row['OpnIntrst']),
                    'oi_change': self._safe_float(row['ChngInOpnIntrst']),
                    'oi_change_pct': self._safe_float(row['pct_ChngInOpnIntrst'])
                }
        
        return {
            'current_price': current_price,
            'strikes_data': strikes_analysis
        }

    def analyze_max_oi_positions(self):
        """Analyze maximum OI positions and changes"""
        if self.oi_table.empty:
            return {}
            
        analysis = {}
        for _, row in self.oi_table.iterrows():
            label_type = str(row.get(self.symbol, '')).strip()
            if label_type and label_type != 'Total':
                analysis[label_type] = {
                    'strike': self._safe_float(row['StrkPric']),
                    'option_type': row.get('OptnTp', ''),
                    'oi': self._safe_float(row['OpnIntrst']),
                    'oi_change': self._safe_float(row['ChngInOpnIntrst']),
                    'oi_change_pct': self._safe_float(row['pct_ChngInOpnIntrst']),
                    'price_change_pct': self._safe_float(row.get('pct_ChngInOptnPric', 0))
                }
        
        return analysis

    def calculate_market_sentiment(self):
        """Calculate market sentiment indicators"""
        # Determine sentiment based on PCR
        if self.pcr > 1.3:
            sentiment = "Bearish"
            sentiment_strength = "Strong" if self.pcr > 1.5 else "Moderate"
        elif self.pcr < 0.7:
            sentiment = "Bullish"
            sentiment_strength = "Strong" if self.pcr < 0.5 else "Moderate"
        else:
            sentiment = "Neutral"
            sentiment_strength = "Balanced"
        
        return {
            'put_call_ratio': self.pcr,
            'sentiment': sentiment,
            'sentiment_strength': sentiment_strength,
            'total_call_oi': self.total_call_oi,
            'total_put_oi': self.total_put_oi,
            'total_oi': self.total_oi
        }

    def generate_commentary(self):
        """Generate comprehensive market commentary"""
        # Run all analyses
        futures_analysis = self.analyze_futures_market()
        options_analysis = self.analyze_options_activity()
        max_oi_analysis = self.analyze_max_oi_positions()
        sentiment_analysis = self.calculate_market_sentiment()
        
        commentary = {
            'market_structure': self._generate_market_structure_commentary(futures_analysis, options_analysis),
            'futures_activity': self._generate_futures_commentary(futures_analysis),
            'options_activity': self._generate_options_commentary(options_analysis, max_oi_analysis),
            'market_sentiment_and_trading': self._generate_sentiment_and_trading_commentary(
                sentiment_analysis, futures_analysis, max_oi_analysis, options_analysis)
        }
        
        return commentary

    def _generate_market_structure_commentary(self, futures_analysis, options_analysis):
        """Generate market structure commentary"""
        commentary = []
        
        current_price = options_analysis.get('current_price', self.curr_close)
        price_change = current_price - self.prev_close
        price_change_pct = (price_change / self.prev_close) * 100 if self.prev_close != 0 else 0
        
        commentary.append(f"Current market price at {current_price:,.2f} ({price_change:+.2f} points, {price_change_pct:+.2f}%)")
        
        if futures_analysis.get('near_month_data'):
            nm_data = futures_analysis['near_month_data']
            fut_price_change = nm_data.get('price_change_pct', 0)
            commentary.append(f"Futures trading at {nm_data.get('settlement_price', 0):,.2f} with {fut_price_change:+.2f}% change")
        
        return commentary

    def _generate_futures_commentary(self, futures_analysis):
        """Generate futures activity commentary"""
        commentary = []
        
        if not futures_analysis:
            commentary.append("No futures data available for analysis")
            return commentary
        
        total_oi = futures_analysis.get('total_oi', 0)
        oi_change = futures_analysis.get('oi_change', 0)
        oi_change_pct = futures_analysis.get('oi_change_pct', 0)
        
        commentary.append(f"Total futures OI: {total_oi:,.0f} contracts ({oi_change:+,.0f}, {oi_change_pct:+.2f}%)")
        
        if futures_analysis.get('near_month_data'):
            nm_data = futures_analysis['near_month_data']
            nm_oi_change = nm_data.get('oi_change', 0)
            nm_oi_change_pct = nm_data.get('oi_change_pct', 0)
            price_change_pct = nm_data.get('price_change_pct', 0)
            
            commentary.append(f"Near month futures OI change: {nm_oi_change:+,.0f} contracts ({nm_oi_change_pct:+.2f}%)")
            
            # Interpret OI vs Price movement
            if price_change_pct > 0 and nm_oi_change > 0:
                commentary.append("Rising prices with increasing OI suggests fresh long buildup")
            elif price_change_pct > 0 and nm_oi_change < 0:
                commentary.append("Rising prices with declining OI indicates short covering")
            elif price_change_pct < 0 and nm_oi_change > 0:
                commentary.append("Falling prices with increasing OI suggests fresh short buildup")
            elif price_change_pct < 0 and nm_oi_change < 0:
                commentary.append("Falling prices with declining OI indicates long unwinding")
        
        return commentary

    def _generate_options_commentary(self, options_analysis, max_oi_analysis):
        """Generate options activity commentary"""
        commentary = []
        
        # Maximum OI positions
        if 'Max Call OI' in max_oi_analysis:
            max_call = max_oi_analysis['Max Call OI']
            commentary.append(f"Maximum CALL OI at {max_call['strike']:.0f} strike ({max_call['oi']:,.0f} contracts)")
        
        if 'Max Put OI' in max_oi_analysis:
            max_put = max_oi_analysis['Max Put OI']
            commentary.append(f"Maximum PUT OI at {max_put['strike']:.0f} strike ({max_put['oi']:,.0f} contracts)")
        
        # OI Changes
        significant_changes = []
        
        if 'Max Put OI addition' in max_oi_analysis:
            put_add = max_oi_analysis['Max Put OI addition']
            if put_add['oi_change'] > 0:
                significant_changes.append(f"PUT buildup at {put_add['strike']:.0f} (+{put_add['oi_change']:,.0f} contracts)")
        
        if 'Max Call OI addition' in max_oi_analysis:
            call_add = max_oi_analysis['Max Call OI addition']
            if call_add['oi_change'] > 0:
                significant_changes.append(f"CALL buildup at {call_add['strike']:.0f} (+{call_add['oi_change']:,.0f} contracts)")
        
        if 'Max Call OI unwinding' in max_oi_analysis:
            call_unwind = max_oi_analysis['Max Call OI unwinding']
            if call_unwind['oi_change'] < 0:
                significant_changes.append(f"CALL unwinding at {call_unwind['strike']:.0f} ({call_unwind['oi_change']:,.0f} contracts)")
        
        if 'Max Put OI unwinding' in max_oi_analysis:
            put_unwind = max_oi_analysis['Max Put OI unwinding']
            if put_unwind['oi_change'] < 0:
                significant_changes.append(f"PUT unwinding at {put_unwind['strike']:.0f} ({put_unwind['oi_change']:,.0f} contracts)")
        
        if significant_changes:
            commentary.append("Significant OI changes: " + ", ".join(significant_changes))
        
        return commentary

    def _generate_sentiment_and_trading_commentary(self, sentiment_analysis, futures_analysis, max_oi_analysis, options_analysis):
        """Generate combined sentiment and trading implications commentary"""
        commentary = []
        
        # Sentiment Analysis
        pcr = sentiment_analysis.get('put_call_ratio', 0)
        sentiment = sentiment_analysis.get('sentiment', 'Neutral')
        sentiment_strength = sentiment_analysis.get('sentiment_strength', 'Balanced')
        
        commentary.append(f"Put-Call ratio: {pcr:.2f} indicates {sentiment_strength.lower()} {sentiment.lower()} sentiment")
        
        if pcr > 1.3:
            commentary.append("High PUT OI suggests bearish positioning, potential support if unwound")
        elif pcr < 0.7:
            commentary.append("High CALL OI suggests bullish positioning, potential resistance if unwound")
        else:
            commentary.append("Balanced PUT-CALL positioning indicates neutral market outlook")
        
        # Key levels and trading implications
        support_levels = []
        resistance_levels = []
        
        current_price = options_analysis.get('current_price', self.curr_close)
        
        # Identify key levels from max OI
        if 'Max Put OI' in max_oi_analysis:
            support_level = max_oi_analysis['Max Put OI']['strike']
            support_levels.append(support_level)
            commentary.append(f"Key support at {support_level:.0f} (Max PUT OI)")
        
        if 'Max Call OI' in max_oi_analysis:
            resistance_level = max_oi_analysis['Max Call OI']['strike']
            resistance_levels.append(resistance_level)
            commentary.append(f"Key resistance at {resistance_level:.0f} (Max CALL OI)")
        
        # Trading implications based on OI analysis
        oi_change_pct = futures_analysis.get('oi_change_pct', 0)
        nm_data = futures_analysis.get('near_month_data', {})
        price_change_pct = nm_data.get('price_change_pct', 0)
        
        if abs(price_change_pct) > 1:  # Significant price movement
            if price_change_pct > 0 and oi_change_pct < -5:
                commentary.append("Rising prices with declining OI suggests caution - possible short covering rally")
            elif price_change_pct < 0 and oi_change_pct > 5:
                commentary.append("Falling prices with rising OI indicates strong bearish conviction")
            elif price_change_pct > 0 and oi_change_pct > 5:
                commentary.append("Rising prices with increasing OI suggests sustained bullish momentum")
        
        # Range-bound trading expectations
        if support_levels and resistance_levels:
            support = min(support_levels)
            resistance = max(resistance_levels)
            if current_price > support and current_price < resistance:
                commentary.append(f"Expect range-bound trading between {support:.0f}-{resistance:.0f}")
                commentary.append(f"Break above {resistance:.0f} or below {support:.0f} could see acceleration")
        
        return commentary

    def _safe_float(self, value):
        """Safely convert value to float"""
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        try:
            if isinstance(value, str):
                value = value.replace(',', '').replace('%', '')
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def print_analysis(self):
        """Print comprehensive analysis"""
        commentary = self.generate_commentary()
        
        print("=" * 80)
        print(f"FINANCIAL MARKET DATA ANALYSIS - {self.symbol}")
        print("=" * 80)
        print(f"Current Price: {self.curr_close:,.2f} | Previous Close: {self.prev_close:,.2f}")
        print(f"Expiry Date: {self.month_expry.strftime('%d-%b-%Y')}")
        print("=" * 80)
        
        print("\n📊 MARKET STRUCTURE:")
        for item in commentary['market_structure']:
            print(f"  • {item}")
        
        print("\n⚡ FUTURES ACTIVITY:")
        for item in commentary['futures_activity']:
            print(f"  • {item}")
        
        print("\n📈 OPTIONS ACTIVITY:")
        for item in commentary['options_activity']:
            print(f"  • {item}")
        
        print("\n🎭 MARKET SENTIMENT & TRADING IMPLICATIONS:")
        for item in commentary['market_sentiment_and_trading']:
            print(f"  • {item}")
        
        print("\n" + "=" * 80)
        
        return commentary

    def get_html_report(self):
        """Return HTML report with inline styling only"""
        def fmt_num(x, d=2):
            try:
                return f"{float(x):,.{d}f}"
            except Exception:
                return str(x)

        def fmt_pct(x, d=2):
            try:
                return f"{float(x):.{d}f}%"
            except Exception:
                return str(x)

        def li_block(items):
            return "".join(f"<li>{str(it)}</li>" for it in items or [])

        summary = self.get_summary_stats()
        commentary = self.generate_commentary()

        curr = summary.get("current_price")
        prev = summary.get("previous_price")
        px_chg = summary.get("price_change", 0)
        px_chg_pct = summary.get("price_change_pct", 0)
        expiry = summary.get("expiry_date")
        expiry_str = expiry.strftime("%d-%b-%Y") if hasattr(expiry, "strftime") else str(expiry)

        sentiment = summary.get("sentiment", "-")
        pcr = summary.get("put_call_ratio")
        tot_opt_oi = summary.get("total_options_oi")
        fut_oi_total = summary.get("futures_oi_total")

        curr_atm_oi_html = df_to_styled_html(self.curr_atm_oi)
        prev_atm_oi_html = df_to_styled_html(self.prev_atm_oi)
        oi_table_html = df_to_styled_html(self.oi_table)
        fut_oi_table_html = df_to_styled_html(self.fut_oi_table)

        chg_dir = "▲" if px_chg >= 0 else "▼"
        chg_color = "#16a34a" if px_chg >= 0 else "#dc2626"

        return f"""<title>Market Report - {self.symbol}</title></head>
<body style="font-family:Arial, sans-serif; background:#f9fafb; color:#111; margin:0; padding:20px;">
  <div style="max-width:900px; margin:0 auto; background:#fff; padding:20px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
    <h2 style="margin-top:0; border-bottom:2px solid #e5e7eb; padding-bottom:10px;">
      {self.symbol}
    </h2>
    <p><b>Current Price:</b> ₹{fmt_num(curr)} <span style="color:{chg_color};">{chg_dir} {fmt_num(px_chg)} ({fmt_pct(px_chg_pct)})</span></p>
    <p><b>Previous Close:</b> ₹{fmt_num(prev)} | <b>Expiry:</b> {expiry_str}</p>

    <h3 style="margin-top:20px; color:#1f2937;">📊 Market Structure</h3>
    <ul style="margin:0; padding-left:20px;">{li_block(commentary.get("market_structure"))}</ul>

    <h3 style="margin-top:20px; color:#1f2937;">⚡ Futures Activity</h3>
    <ul style="margin:0; padding-left:20px;">{li_block(commentary.get("futures_activity"))}</ul>

    <h3 style="margin-top:20px; color:#1f2937;">📈 Options Activity</h3>
    <ul style="margin:0; padding-left:20px;">{li_block(commentary.get("options_activity"))}</ul>

    <h3 style="margin-top:20px; color:#1f2937;">🎭 Market Sentiment & Trading Implications</h3>
    <div style="padding:10px; border-radius:6px; background:#f3f4f6; margin-bottom:10px; font-weight:bold; color:#374151;">
      Sentiment: {sentiment} | Put-Call Ratio: {fmt_num(pcr,2) if pcr else "-"}
    </div>
    <ul style="margin:0; padding-left:20px;">{li_block(commentary.get("market_sentiment_and_trading"))}</ul>

    <h3 style="margin-top:20px; color:#1f2937;">📌 Summary Stats</h3>
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
      <tr style="background:#f9fafb;">
        <th style="text-align:left; padding:8px; border:1px solid #e5e7eb;">Previous Close</th>
        <td style="padding:8px; border:1px solid #e5e7eb;">₹{fmt_num(prev)}</td>
      </tr>
      <tr>
        <th style="text-align:left; padding:8px; border:1px solid #e5e7eb;">Total Options OI</th>
        <td style="padding:8px; border:1px solid #e5e7eb;">{fmt_num(tot_opt_oi,0)}</td>
      </tr>
      <tr style="background:#f9fafb;">
        <th style="text-align:left; padding:8px; border:1px solid #e5e7eb;">Total Futures OI</th>
        <td style="padding:8px; border:1px solid #e5e7eb;">{fmt_num(fut_oi_total,0)}</td>
      </tr>
    </table>

    <h3 style="margin-top:20px; color:#1f2937;">📊 Futures OI Table</h3>
    <div style="overflow-x:auto; margin-bottom:20px;">
      {fut_oi_table_html}
    </div> 
    <h3 style="margin-top:20px; color:#1f2937;">📈 Options OI Table</h3>
    <div style="overflow-x:auto; margin-bottom:20px;">
      {oi_table_html}
    </div>
    <h3 style="margin-top:20px; color:#1f2937;">📈 Current ATM OI</h3>
    <div style="overflow-x:auto; margin-bottom:20px;">
      {curr_atm_oi_html}
    </div>
    <h3 style="margin-top:20px; color:#1f2937;">📈Previous ATM OI</h3>
    <div style="overflow-x:auto; margin-bottom:20px;">
      {prev_atm_oi_html}
    </div>

  </div>
    """
    
    
    
    def get_summary_stats(self):
        """Get summary statistics"""
        sentiment_data = self.calculate_market_sentiment()
        return {
            'symbol': self.symbol,
            'current_price': self.curr_close,
            'previous_price': self.prev_close,
            'price_change': self.curr_close - self.prev_close,
            'price_change_pct': ((self.curr_close - self.prev_close) / self.prev_close) * 100 if self.prev_close != 0 else 0,
            'expiry_date': self.month_expry,
            'futures_oi_total': self._safe_float(self.fut_oi_table.iloc[-1]['OpnIntrst']) if not self.fut_oi_table.empty else 0,
            'sentiment': sentiment_data.get('sentiment', 'Unknown'),
            'put_call_ratio': sentiment_data.get('put_call_ratio', 0),
            'total_options_oi': sentiment_data.get('total_oi', 0)
        }

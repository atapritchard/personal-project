#############################
# Trade analyses
# Alden Pritchard
# Quarantine 2020
#############################

import csv
import datetime
import functools
from tqdm import tqdm
from exfil_algos import apply_stop_exit as exit_position
import numpy as np
from pdb import set_trace as debug


# Add strategy tracking over time, will be complicated
def run_analysis(symbol, pct_dn=None, pct_up=None, n=None, verbose=False, identifier='', dest='exfil'):
    # Read in the dataset
    with open('{1}/tradefiles/{0}.csv'.format(symbol, dest), 'r', newline='') as file:
        trades = list(filter(lambda x: '15:30' not in x[0], list(csv.reader(file, delimiter=','))))
    header, rule_str = trades.pop(0), '{0}x{1}'.format(pct_dn, pct_up)

    # Compute performances under exit rule
    date_converter = lambda dt_str: datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    trades = list(map(lambda row: [date_converter(row[0])] + row[1:], trades))
    # debug()
    updated_trades = exit_position(symbol, trades, pct_dn, pct_up, n)
    with open('{0}/trade_log.csv'.format(dest), 'a', newline='') as file:
        csv.writer(file, delimiter=',').writerows(
            list(map(lambda row: [str(row[0]), symbol, rule_str] + row[1:], updated_trades)))

    # Compute analysis metrics
    perfs = list(map(lambda x: 1 + x[-2] / 100, updated_trades))
    performance = (functools.reduce(lambda x, y: x * y, perfs) - 1) * 100
    acc = len(list(filter(lambda x: x >= 1, perfs))) / len(perfs) * 100
    avg_ticks = sum(list(map(lambda x: x[-1], updated_trades))) / len(updated_trades)

    # Output results, if desired
    if verbose:
        print(symbol)
        print('Rule:', '{0} x {1}'.format(pct_dn, pct_up))
        print('Performance: ', round(performance, 2), '%', sep="")
        print('Accuracy: ', round(acc, 1), '%', sep="")
        print('Avg # Ticks:', round(avg_ticks, 1))
        print('# Trades:', len(perfs))

    return symbol, '{0}x{1}'.format(pct_dn, pct_up), pct_dn, pct_up, performance, acc, avg_ticks, len(perfs)


def run_analyses(symbols, rules, tag='', n_ticks=6, dest='ml_research'):
    # Clear trade log
    with open('{0}/trade_log.csv'.format(dest), 'w', newline='') as file:
        csv.writer(file, delimiter=',').writerows(
            [['Datetime', 'Symbol', 'Rule', 'Side', 'Performance', 'NumTicks']])

    # Run analyses and collect summary results
    runs = []
    for symbol in symbols:
        for pct_dn, pct_up in rules:
            runs.append((symbol, pct_dn, pct_up))
    runs.sort(key=lambda run: '{0}x{1}'.format(run[1], run[2]))
    output = list([['Symbol', 'Rule', 'PctDown', 'PctUp', 'Performance', 'Accuracy', 'AvgNumTicks', 'NumTrades']])
    for sym, dn, up in tqdm(runs, desc='Running analyses', ncols=80):
        output.append(run_analysis(sym, pct_up=up, pct_dn=dn, identifier=tag, n=n_ticks, dest=dest))

    # Save summary results to file
    with open('{0}/analyses.csv'.format(dest, tag), 'w', newline='') as file:
        csv.writer(file, delimiter=',').writerows(output)


def main(symbols):
    run_analyses(symbols, [(0.2, 0.4), (0.5, 2), (3, 3)])


if __name__ == '__main__':
    raise NotImplemented
    # main()

########################################################################################################################
# R&D for ML methods
# Alden Pritchard
# October 2020
########################################################################################################################


import ml_research.dataset_builder as dsb
import ml_research.modeling as ml_rnd
import ml_research.trade_analysis as tal


def main():
    # symbols, symbol_set = ['JPM', 'GS', 'BAC', 'C'], 'banks'
    # symbols, symbol_set = ['MU', 'LRCX', 'AMAT'], 'semis'
    symbols, symbol_set = ['MU', 'LRCX', 'AMAT'] + ['JPM', 'GS', 'BAC', 'C'], 'hybrid'

    dsb.main(symbols, symbol_set)
    ml_rnd.main(symbol_set)
    tal.main(symbols)


if __name__ == '__main__':
    main()

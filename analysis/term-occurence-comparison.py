from matplotlib import pyplot as plt
import pandas as pd

output_file_name = './data/pp-data-1732556014.112211.csv'

df = pd.read_csv(output_file_name)

df['year'] = df['year'].astype(int)

# aggregate the number of occurences of the word 'case' in title column and aggregate by year
case_counts = df['title'].str.contains('case', case=False).groupby(df['year']).sum()
blockchain_counts = df['title'].str.contains('blockchain', case=False).groupby(df['year']).sum()

# Create a bar chart with the counts
plt.figure(figsize=(10, 6))
plt.bar(case_counts.index, case_counts.values, label='Case')
plt.bar(blockchain_counts.index, blockchain_counts.values, label='Blockchain')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of Occurrences of "Case" and "Blockchain" in Title Column per Year')
plt.legend()
plt.savefig("./output/sample/term-occurence-comparison-blockchain-vs-case.png")

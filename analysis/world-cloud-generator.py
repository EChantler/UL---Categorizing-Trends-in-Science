from matplotlib import gridspec, pyplot as plt
import pandas
from wordcloud import WordCloud


def GetWordCloud(cluster_text):
    cluster_text = cluster_text.split('\n')
    cluster_text = cluster_text[:-1]
    cluster_text = [line.split(': ') for line in cluster_text]

    cluster_text = {line[0]: float(line[1]) for line in cluster_text}
    print(cluster_text)

    # generate word cloud

    wordcloud = WordCloud(
        width=800,                # Width of the canvas
        height=400,               # Height of the canvas
        background_color='white', # Background color
        colormap='viridis'        # Color scheme
    ).generate_from_frequencies(cluster_text)
    return wordcloud

cluster_0_text = '''-> categories_astro-ph: 0.2348
 -> categories_physics: 0.1715
 -> categories_cond-mat: 0.1352
 -> categories_gr-qc: 0.1084
 -> word_dark matter: 0.0953
 -> word_black hole: 0.0807
 -> word_black holes: 0.0766
 -> categories_hep-ph: 0.0731
 -> categories_hep-th: 0.0643
  word_machine learning: 0.0615
 -> categories_quant-ph: 0.0558
 -> categories_q-bio: 0.0481
 -> word_gravitational wave: 0.0422
 -> word_magnetic field: 0.0342
 -> word_field theory: 0.0338
 -> word_gravitational waves: 0.0317
  word_neural networks: 0.0293
 -> word_phase transition: 0.0287
  word_covid 19: 0.0284
  word_deep learning: 0.0279'''

cluster_1_text = '''-> categories_cs: 1.0000
 -> word_language models: 0.0643
  word_reinforcement learning: 0.0642
  word_neural networks: 0.0585
  word_deep learning: 0.0526
  word_machine learning: 0.0490
 -> word_large language: 0.0412
 -> word_large language models: 0.0357
  word_neural network: 0.0347
  word_federated learning: 0.0333
  word_self supervised: 0.0221
  word_multi agent: 0.0217
  word_large scale: 0.0214
  word_real time: 0.0201
 -> word_language model: 0.0183
 -> word_object detection: 0.0181
  word_representation learning: 0.0177
  word_covid 19: 0.0174
  word_learning based: 0.0172
  word_time series: 0.0166'''
 
cluster_2_text ='''-> categories_math: 1.0000
 -> word_differential equations: 0.0813
 -> word_optimal control: 0.0555
 -> word_mean field: 0.0537
  word_neural networks: 0.0525
 -> word_dynamical systems: 0.0454
 -> word_finite element: 0.0424
  word_high dimensional: 0.0423
 -> word_second order: 0.0412
 -> word_higher order: 0.0368
  word_optimal transport: 0.0319
  word_massive mimo: 0.0302
  word_data driven: 0.0299
 -> word_high order: 0.0286
  word_machine learning: 0.0256
  word_gradient descent: 0.0249
  word_deep learning: 0.0248
  word_neural network: 0.0234
  word_reinforcement learning: 0.0234
  word_low rank: 0.0232'''
 
cluster_3_text ='''-> categories_stat: 1.0000
  word_neural networks: 0.1022
  word_high dimensional: 0.0967
  word_time series: 0.0807
  word_machine learning: 0.0769
  word_reinforcement learning: 0.0533
  word_covid 19: 0.0463
  word_deep learning: 0.0459
 -> word_monte carlo: 0.0388
  word_neural network: 0.0335
  word_gradient descent: 0.0235
  word_large scale: 0.0199
  word_optimal transport: 0.0191
  word_federated learning: 0.0182
  word_low rank: 0.0175
  word_data driven: 0.0175
 -> word_spatio temporal: 0.0172
 -> word_deep neural: 0.0170
 -> word_model based: 0.0156
  word_representation learning: 0.0153'''
 
cluster_4_text ='''-> categories_eess: 1.0000
  word_deep learning: 0.0914
  word_neural network: 0.0502
  word_reinforcement learning: 0.0476
  word_neural networks: 0.0446
 -> word_speech recognition: 0.0421
  word_real time: 0.0396
  word_learning based: 0.0385
 -> word_end end: 0.0375
  word_machine learning: 0.0367
  word_data driven: 0.0367
 -> word_super resolution: 0.0334
  word_self supervised: 0.0331
  word_massive mimo: 0.0324
  word_covid 19: 0.0254
  word_large scale: 0.0214
 -> word_convolutional neural: 0.0209
  word_multi agent: 0.0205
 -> word_using deep: 0.0204
 -> word_generative adversarial: 0.0186'''

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2,1, hspace=0.0, wspace=0.0)

gs0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0])
gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])

ax1 = fig.add_subplot(gs0[0])
ax1.imshow(GetWordCloud(cluster_0_text), interpolation='bilinear')
ax1.set_title('Cluster 0')
ax1.axis('off')

ax2 = fig.add_subplot(gs0[1])
ax2.imshow(GetWordCloud(cluster_1_text), interpolation='bilinear')
ax2.set_title('Cluster 1')
ax2.axis('off')

ax3 = fig.add_subplot(gs0[2])
ax3.imshow(GetWordCloud(cluster_2_text), interpolation='bilinear')
ax3.set_title('Cluster 2')
ax3.axis('off')

# Plot 2 graphs in the second row
ax4 = fig.add_subplot(gs1[0])
ax4.imshow(GetWordCloud(cluster_3_text), interpolation='bilinear')
ax4.set_title('Cluster 3')
ax4.axis('off')

ax5 = fig.add_subplot(gs1[1])
ax5.imshow(GetWordCloud(cluster_4_text), interpolation='bilinear')
ax5.set_title('Cluster 4')
ax5.axis('off')

fig.tight_layout()

plt.savefig(f"output/k-means-clusters-wordcloud-2020-2024.png")
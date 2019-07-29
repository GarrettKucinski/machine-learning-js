require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCsv = require('./load-csv')

const options = {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living', 'condition'],
  labelColumns: ['price']
}

let {
  features,
  labels,
  testFeatures,
  testLabels
} = loadCsv('kc_house_data.csv', options)

function knn (features, labels, predictionPoint, k) {
  // standardization value - average / sqrt of variance
  const { mean, variance } = tf.moments(features, 0)
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

  return features
    .sub(mean)
    .div(variance.pow(0.5))
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
    .slice(0, k)
    .reduce((acc, t) => acc + t.get(1), 0) / k
}

features = tf.tensor(features)
labels = tf.tensor(labels)

const k = 20

testFeatures.forEach((feature, i) => {
  const testLabel = testLabels[i]

  const result = knn(features, labels, tf.tensor(feature), k)
  const err = (testLabel - result) / testLabel

  console.log('guess', result, testLabel)
  console.log('percentage of error', err * 100)
})

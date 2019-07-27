let outputs = []

const predictionPoint = 300
const k = 6

const ascending = (a, b) => a[1] > b[1]
const descending = (a, b) => a[1] > b[1]

function shuffle (a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function splitDataSet (data, countToTest) {
  const shuffled = shuffle(data)
  const testSet = shuffled.slice(0, countToTest)
  const trainingSet = shuffled.slice(countToTest)

  return [testSet, trainingSet]
}

function sortBy (direction) {
  return function (a, b) {
    if (direction(a, b)) return -1
    if (!direction(a, b)) return 1
    return 0
  }
}

function createPairs (obj) {
  return Object.entries(obj).map(([key, value]) => [key, value])
}

function distance (point) {
  return Math.abs(point - predictionPoint)
}

function calculateAbsoluteDropPosition (data) {
  const [position, , , bucket] = data
  return [distance(position), bucket]
}

function findMostCommonRecord (acc, [point, bucket]) {
  if (acc[bucket]) {
    acc[bucket] += 1
  } else {
    acc[bucket] = 1
  }

  return acc
}

function onScoreUpdate (dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs = [...outputs, [dropPosition, bounciness, size, bucketLabel]]
}

function runAnalysis () {
  // Write code here to analyze stuff

  const positions = outputs.map(calculateAbsoluteDropPosition)
  const sorted = positions.sort(sortBy(ascending))
  const map = sorted.reduce(findMostCommonRecord, {})
  const result = createPairs(map).sort(sortBy(descending)).slice(0, k)

  const [[bucket]] = result

  console.log('It will probably fall into', bucket)
}

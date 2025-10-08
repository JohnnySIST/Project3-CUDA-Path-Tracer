option = {
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      // Use axis to trigger tooltip
      type: 'shadow' // 'shadow' as default; can also be 'line' or 'shadow'
    }
  },
  legend: {},
  xAxis: {
    type: 'value'
  },
  yAxis: {
    type: 'category',
    data: ['Open', 'Closed']
  },
  series: [
    {
      name: 'generateRayFromCamera',
      type: 'bar',
      stack: 'total',
      label: {
        show: true
      },
      emphasis: {
        focus: 'series'
      },
      data: [1.022, 1.022]
    },
    {
      name: 'computeIntersections',
      type: 'bar',
      stack: 'total',
      label: {
        show: true
      },
      emphasis: {
        focus: 'series'
      },
      data: [4.631, 10.503]
    },
    {
      name: 'shadeBSDFMaterial',
      type: 'bar',
      stack: 'total',
      label: {
        show: true
      },
      emphasis: {
        focus: 'series'
      },
      data: [0.614, 0.779]
    },
  ]
};
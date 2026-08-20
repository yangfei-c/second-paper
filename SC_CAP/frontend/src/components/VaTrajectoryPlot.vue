<script setup>
import { computed } from "vue";
import { buildTrajectoryPlot } from "../utils/vaPlot";

const props = defineProps({ planned: { type: Array, default: () => [] }, actual: { type: Array, default: () => [] } });
const plot = computed(() => buildTrajectoryPlot(props.planned, props.actual));
</script>

<template>
  <div class="va-plot-wrap">
    <svg class="va-plot" viewBox="0 0 220 160" role="img" aria-label="计划音乐轨迹与用户真实轨迹">
      <line v-if="plot.axis.showY" x1="20" :y1="plot.axis.y" x2="200" :y2="plot.axis.y" />
      <line v-if="plot.axis.showX" :x1="plot.axis.x" y1="20" :x2="plot.axis.x" y2="140" />
      <polyline v-if="plot.plannedPolyline" class="planned-line" :points="plot.plannedPolyline" />
      <polyline v-if="plot.actualPolyline" class="actual-line" :points="plot.actualPolyline" />
      <g v-for="node in plot.nodes" :key="`${node.type}-${node.label}-${node.x}`">
        <circle :class="node.type" :cx="node.x" :cy="node.y" r="5" />
        <text :x="node.x + 7" :y="node.y - 7">{{ node.label }}</text>
      </g>
    </svg>
    <div class="plot-legend">
      <span><i class="text"></i>文本预测</span><span><i class="music"></i>音乐预测</span>
      <span><i class="user"></i>用户初始</span><span><i class="felt"></i>用户感受</span>
    </div>
  </div>
</template>

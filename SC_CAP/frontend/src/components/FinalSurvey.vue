<script setup>
const ratings = defineModel("ratings");
defineProps({ loading: Boolean });
defineEmits(["submit"]);
const fields = [["overall_strategy_fit", "整体策略契合度"], ["satisfaction", "总体满意度"], ["enjoyment", "聆听愉悦度"], ["smoothness", "音乐衔接平滑度"], ["willingness_to_use_again", "再次使用意愿"]];
</script>

<template>
  <div class="modal-backdrop"><form class="modal" @submit.prevent="$emit('submit')"><h2>总体评价</h2><div class="survey-grid"><label v-for="[key, label] in fields" :key="key">{{ label }}<select v-model="ratings[key]"><option :value="1">1 很低</option><option :value="2">2 较低</option><option :value="3">3 一般</option><option :value="4">4 较高</option><option :value="5">5 很高</option></select></label></div><button class="primary-button" :disabled="loading">{{ loading ? "保存中..." : "保存本次记录" }}</button></form></div>
</template>

import { ref } from "vue";

export function usePlayback() {
  const duration = ref(0);
  const repeatCount = ref(0);
  const ended = ref(false);

  function reset() {
    duration.value = 0;
    repeatCount.value = 0;
    ended.value = false;
  }
  function play(event) {
    if (ended.value) {
      repeatCount.value += 1;
      ended.value = false;
    }
  }
  function timeupdate(event) {
    duration.value = Math.max(duration.value, event?.target?.currentTime || 0);
  }
  function finish(event) {
    timeupdate(event);
    ended.value = true;
  }
  return { duration, repeatCount, ended, reset, play, timeupdate, finish };
}

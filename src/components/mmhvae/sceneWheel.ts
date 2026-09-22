/** Labels sit above the canvas but belong to the same zoomable scene. */
export function forwardSceneWheel(labels:HTMLElement,canvas:HTMLCanvasElement){
 labels.addEventListener('wheel',event=>{event.preventDefault();canvas.dispatchEvent(new WheelEvent('wheel',{deltaX:event.deltaX,deltaY:event.deltaY,deltaMode:event.deltaMode,clientX:event.clientX,clientY:event.clientY,ctrlKey:event.ctrlKey,shiftKey:event.shiftKey,cancelable:true}))},{passive:false});
}

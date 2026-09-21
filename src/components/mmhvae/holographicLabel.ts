import * as T from 'three'

/** World-space typography: a quiet luminous plate, physically anchored to its layer. */
export function holographicLabel(title:string,subtitle='',width=5.5){
 const group=new T.Group(),canvas=document.createElement('canvas');canvas.width=768;canvas.height=160;
 const ctx=canvas.getContext('2d')!;
 ctx.clearRect(0,0,768,160);ctx.fillStyle='rgba(6,18,28,.72)';ctx.fillRect(0,0,768,160);
 ctx.strokeStyle='#639bb4';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(1,142);ctx.lineTo(1,20);ctx.lineTo(30,20);ctx.stroke();
 ctx.fillStyle='#d4f4fa';ctx.font='500 44px "Microsoft YaHei UI",sans-serif';
 let text=title;while(ctx.measureText(text).width>715&&text.length>1)text=text.slice(0,-1);if(text!==title)text+='…';ctx.fillText(text,25,65);
 ctx.font='30px "Microsoft YaHei UI",sans-serif';ctx.fillStyle='#95bdd0';ctx.fillText(subtitle,25,119,710);
 const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;
 const plate=new T.Mesh(new T.PlaneGeometry(width,width*160/768),new T.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false,side:T.DoubleSide,toneMapped:false}));
 plate.position.x=width/2;group.add(plate);group.userData.holographicLabel=true;group.userData.title=title;group.userData.width=width;
 return group;
}

/** Keep world-anchored lettering legible under perspective; unlike a DOM
 * tooltip it still rotates, occludes and moves with the model. */
export function sizeHolographicLabel(group:T.Group,camera:T.PerspectiveCamera,height:number,pixels=220){
 const distance=camera.position.distanceTo(group.getWorldPosition(new T.Vector3()));
 const units=2*Math.tan(T.MathUtils.degToRad(camera.fov/2))*distance/(height*camera.zoom);
 group.scale.setScalar(Math.max(.7,pixels*units/group.userData.width));
}

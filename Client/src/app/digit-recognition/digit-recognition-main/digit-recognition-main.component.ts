import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import {Position, TouchService} from "../../touch.service";
import {DrawerService} from "../../drawer.service";
import {PythonRunnerService} from "../../python-runner.service";
import {AiReposService} from "../../ai-repos.service";

@Component({
    selector: 'app-digit-recognition-main',
    templateUrl: './digit-recognition-main.component.html',
    styleUrls: ['./digit-recognition-main.component.scss']
})
export class DigitRecognitionMainComponent implements AfterViewInit {
    @ViewChild("canvasElement")
    private canvasElement!: ElementRef<HTMLCanvasElement>;
    
    @ViewChild("resizeCanvasElement")
    private resizeCanvasElement!: ElementRef<HTMLCanvasElement>;
    
    public busy = true;
    public result?: number;
    private prevPos?: Position;
    
    public constructor(private touchService: TouchService,
                       private drawerService: DrawerService,
                       private pythonRunner: PythonRunnerService,
                       private aiReposService: AiReposService) {
    }
    
    public async ngAfterViewInit() {
        let canvas = this.canvasElement.nativeElement;
        let ctx = canvas.getContext("2d")!;
        
        this.touchService.dragEmitter.subscribe(({x, y}) => {
            let currX = x - canvas.offsetLeft - this.drawerService.currentWidth;
            let currY = y - canvas.offsetTop;
            
            if (this.prevPos !== undefined) {
                ctx.beginPath();
                ctx.moveTo(this.prevPos.x, this.prevPos.y);
                ctx.lineTo(currX, currY);
                ctx.lineWidth = 6;
                ctx.strokeStyle = "black";
                ctx.shadowColor = "black";
                ctx.shadowBlur = 4;
                ctx.lineCap = "round";
                ctx.lineJoin = "round";
                ctx.stroke();
                ctx.closePath();
            }
            
            this.prevPos = {x: currX, y: currY};
        });
        
        this.touchService.touchEndEmitter.subscribe(() => {
            this.prevPos = undefined;
        });
        
        await this.pythonRunner.loadScript("digit_recognition");
        
        await this.aiReposService.loadFromServer("digit_recognition", this.pythonRunner);
        
        await this.pythonRunner.run("instance.echo()", {message: "123"});
        console.log("Loaded")
        this.busy = false;
    }
    
    public clear() {
        this.canvasElement.nativeElement.getContext("2d")!.clearRect(0, 0, 112, 112);
    }
    
    public async predict() {
        console.log("Predicting");
        this.busy = true;
        
        let image = this.getResizedImage();
        
        this.result = await this.pythonRunner.run("instance.eval()", {inputs: image});
        
        this.busy = false;
    }
    
    private getResizedImage() {
        let canvas = this.canvasElement.nativeElement;
        
        let resizeCanvas = this.resizeCanvasElement.nativeElement;
        let resizeContext = resizeCanvas.getContext("2d")!;
        
        resizeContext.clearRect(0, 0, 28, 28);
        resizeContext.drawImage(canvas, 0, 0, 28, 28);
        
        let img = resizeContext.getImageData(0, 0, 28, 28, {colorSpace: "srgb"});
        let pixels = img.data.filter((value, index) => index % 4 === 3);
        
        return Array.from(pixels.values()).map(o => o / 255.0);
    }
}

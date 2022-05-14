import {Injectable} from '@angular/core';
import {AngularFireDatabase} from "@angular/fire/compat/database";
import {map, firstValueFrom} from "rxjs";
import {zip} from "./utils";
import {PythonRunnerService} from "./python-runner.service";
import {AngularFireStorage} from "@angular/fire/compat/storage";
import {HttpClient} from "@angular/common/http";
import {Buffer} from "buffer";

export type AiName = "digit_recognition";

export type AiModel = ModelInfo & ModelData;

type ModelInfo = {
    version: number;
    hash?: string;
};

type ModelData = {
    params: number[];
};

type ModelCache = {
    hash: string;
    params: number[];
};

function floatsToBuffer(floats: number[]) {
    return Buffer.from(new Uint8Array(new Float32Array(floats).buffer).buffer);
}

function bufferToFloats(buffer: Buffer) {
    return Array.from(new Float32Array(new Uint8Array(buffer).buffer));
}

async function calcHash(buffer: Uint8Array) {
    let digest = await crypto.subtle.digest("SHA-256", buffer)
    return new Uint8Array(digest).reduce((prev, val) => prev + String.fromCharCode(val), "");
}

async function blobToBuffer(blob: Blob) {
    const fileReader = new FileReader();
    return new Promise<Buffer>((resolve, reject) => {
        fileReader.onloadend = (ev) => {
            resolve(ev.target?.result as Buffer);
        };
        fileReader.onerror = reject;
        fileReader.readAsArrayBuffer(blob);
    });
}

@Injectable({
    providedIn: 'root'
})
export class AiReposService {
    private cached = new Map<AiName, ModelCache>();
    
    public constructor(private readonly db: AngularFireDatabase,
                       private readonly storage: AngularFireStorage,
                       private readonly httpClient: HttpClient) {
    }
    
    public async loadFromServer(name: AiName, pythonRunner: PythonRunnerService) {
        let cachedVersion = this.cached.get(name);
        let serverVersion = await this.loadInfo(name);
        
        console.log(serverVersion, cachedVersion)
        
        if (serverVersion?.hash === undefined) {
            let firstVersion: AiModel = await pythonRunner.run("instance.save()");
            await this.saveFirstVersion(name, firstVersion);
            serverVersion = await this.loadInfo(name);
        }
        
        if (await pythonRunner.run("instance.should_load()", {hash: serverVersion.hash})) {
            console.log("Loading model");
            let modelData = await this.loadModelData(name, serverVersion);
            await pythonRunner.run("instance.load()", {
                version: serverVersion.version,
                hash: serverVersion.hash,
                params: modelData!.params
            });
        } else {
            console.log("Skipped loading model")
        }
    }
    
    private async saveFirstVersion(name: AiName, model: AiModel) {
        let hash = await this.saveModelData(name, {params: model.params});
        await this.saveModelInfo(name, {version: 1, hash: hash});
    }
    
    // TODO: avoid conflicts
    public async saveModelChanges(name: AiName, prevParams: number[], newParams: number[]) {
        const paramsDelta = zip(newParams, prevParams).map(o => o[0] - o[1]);
        
        let modelInfo = await this.loadInfo(name);
        let data = await this.loadModelData(name, modelInfo);
        if (data === null) throw new Error();
        
        let updated = zip(data.params, paramsDelta).map(o => o[0] + o[1]);
        
        modelInfo.version++;
        modelInfo.hash = await this.saveModelData(name, {params: updated});
        
        await this.saveModelInfo(name, modelInfo);
    }
    
    private async loadInfo(name: AiName) {
        const path = AiReposService.createModelPath(name);
        return firstValueFrom(this.db.object<ModelInfo>(path).snapshotChanges().pipe(
            map(o => o.payload.val()!)
        ));
    }
    
    private async saveModelInfo(name: AiName, data: ModelInfo) {
        const path = AiReposService.createModelPath(name);
        await this.db.object<ModelInfo>(path).set(data);
    }
    
    private async loadModelData(name: AiName, currentInfo: ModelInfo): Promise<ModelData | null> {
        // No version available in the server
        if (currentInfo.hash === undefined) return null;
        
        let cached = this.cached.get(name);
        if (cached !== undefined && cached.hash === currentInfo.hash) {
            console.log("Loaded from dict cache");
            return {params: cached.params};
        }
    
        console.log("Loading from server");
        let fileURL = await firstValueFrom(this.storage.ref(AiReposService.createModelFile(name)).getDownloadURL());
        let blob = await firstValueFrom(this.httpClient.get(fileURL, {responseType: "blob"}));
        let buffer = await blobToBuffer(blob);
        
        let params = bufferToFloats(buffer);
        this.cached.set(name, {hash: currentInfo.hash, params: params});
        return {params: params};
    }
    
    private async saveModelData(name: AiName, data: ModelData) {
        let buffer = floatsToBuffer(data.params);
        let hash = await calcHash(buffer);
        
        await this.storage.ref(AiReposService.createModelFile(name)).put(buffer);
        this.cached.set(name, {params: data.params, hash: hash});
        return hash;
    }
    
    private static createModelPath(name: AiName) {
        return "/models/" + name;
    };
    
    private static createModelFile(name: AiName) {
        return "/models/" + name + ".dat";
    };
}

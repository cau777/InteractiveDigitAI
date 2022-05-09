import {Injectable} from '@angular/core';
import {AngularFireDatabase} from "@angular/fire/compat/database";
import {map, firstValueFrom} from "rxjs";
import {zip} from "./utils";
import {PythonRunnerService} from "./python-runner.service";

export type AiName = "digit_recognition";

export type AiModel = {
    version: number;
    params: number[];
}

type AiModelBytes = {
    version: number;
    params: string;
}

function floatsToString(floats: number[]) {
    let bytes = new Uint8Array(new Float32Array(floats).buffer);
    return bytes.reduce((prev, val) => prev + String.fromCharCode(val), "");
}


function stringToFloats(str: string) {
    let bytes = new Uint8Array(str.length);
    for (let i = 0; i < str.length; i++) {
        bytes[i] = str.charCodeAt(i);
    }
    
    return Array.from(new Float32Array(bytes.buffer));
}

function encodeModel(model: AiModel): AiModelBytes {
    return {version: model.version, params: floatsToString(model.params)};
}

function decodeModelBytes(modelBytes: AiModelBytes): AiModel {
    return {version: modelBytes.version, params: stringToFloats(modelBytes.params)};
}


@Injectable({
    providedIn: 'root'
})
export class AiReposService {
    private cached = new Map<AiName, AiModel>();
    
    public constructor(private readonly db: AngularFireDatabase) {
    }
    
    public async loadFromServer(name: AiName, pythonRunner: PythonRunnerService) {
        let cachedVersion = this.cached.get(name);
        let serverVersion = await this.getCurrentVersion(name);
    
        console.log(serverVersion, cachedVersion)
        
        if (serverVersion === null) {
            let firstVersion: AiModel = await pythonRunner.run("instance.save()");
            let succeeded = await this.saveFirstVersion(name, firstVersion);
            if (succeeded) {
                this.cached.set(name, firstVersion);
                return;
            }
            serverVersion = 1;
        }
        
        serverVersion = await this.getCurrentVersion(name);
        if (cachedVersion === undefined || serverVersion! > cachedVersion.version) {
            let read = await this.readModel(name);
            this.cached.set(name, read!);
        }
        
        let model = this.cached.get(name)!;
        await pythonRunner.run("instance.load()", {version: model.version, params: model.params});
    }
    
    private async saveFirstVersion(name: AiName, model: AiModel) {
        const path = AiReposService.createModelPath(name);
        let succeeded = false;
        
        await this.db.object<AiModelBytes>(path).query.ref.transaction(transaction);
        
        function transaction(modelBytes: AiModelBytes): AiModelBytes {
            if (modelBytes !== null) return modelBytes;
            succeeded = true;
            return encodeModel(model);
        }
        
        return succeeded;
    }
    
    private getCurrentVersion(name: AiName) {
        const path = AiReposService.createModelPath(name) + "/version";
        return firstValueFrom(this.db.object<number>(path).snapshotChanges().pipe(
            map(o => o.payload.val())
        ));
    }
    
    public async readModel(name: AiName) {
        const path = AiReposService.createModelPath(name);
        let result = await firstValueFrom(this.db.object<AiModelBytes>(path).snapshotChanges().pipe(
            map(o => {
                console.log("exists", o.payload.exists());
                return decodeModelBytes(o.payload.val()!);
            })
        ));
        this.cached.set(name, result);
        return result;
    }
    
    public async saveModelChanges(name: AiName, prevParams: number[], newParams: number[]) {
        const paramsDelta = zip(newParams, prevParams).map(o => o[0] - o[1]);
        const path = AiReposService.createModelPath(name);
        let saved: AiModel;
        
        await this.db.object<AiModelBytes>(path).query.ref.transaction(transaction);
        
        function transaction(modelBytes: AiModelBytes): AiModelBytes {
            let model = decodeModelBytes(modelBytes);
            let updatedParams = zip(model.params, paramsDelta).map(o => o[0] + o[1]);
    
            saved = {
                version: modelBytes.version + 1,
                params: updatedParams
            };
            
            return encodeModel(saved);
        }
        
        this.cached.set(name, saved!);
    }
    
    private static createModelPath(name: AiName) {
        return "/models/" + name;
    };
}

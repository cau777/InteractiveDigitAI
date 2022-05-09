// noinspection JSFileReferences
/// <reference lib="webworker" />

import {IPyodideInitMessage, IPyodideResultMessage, PyodideWorkerMessage} from "./pyodide-worker-messages";
import {PyodideWorkerLogic} from "./pyodide/pyodide-worker-logic";

importScripts("https://cdn.jsdelivr.net/pyodide/v0.20.0/full/pyodide.js");

let worker: PyodideWorkerLogic | undefined = undefined;
let working: boolean = false;

async function initWorker(data: IPyodideInitMessage) {
    worker = new PyodideWorkerLogic(data.baseUrl);
}

function selectAction(data: PyodideWorkerMessage): Promise<unknown | undefined> {
    if (worker === undefined) {
        if (data.action !== "init") throw new Error("Worker not initialized");
        return initWorker(data);
    }
    
    switch (data.action) {
        case "select":
            return worker.select(data.code, data.data);
        case "run":
            return worker.run(data.expression, data.params);
    }
    throw new RangeError();
}

function postResult(r: IPyodideResultMessage) {
    // console.log("Worker posted result " + JSON.stringify(r));
    postMessage(r);
}

addEventListener("message", ({data}: { data: PyodideWorkerMessage }) => {
    // console.log("Worker received message " + JSON.stringify(data));
    if (working) throw new Error("Worker is already busy");
    
    try {
        let action = selectAction(data);
        
        working = true;
        action.then(r => {
            postResult({action: "result", content: r});
            working = false;
        }, e => {
            working = false;
            throw e;
        });
    } catch (e) {
        working = false;
        throw e;
    }
});

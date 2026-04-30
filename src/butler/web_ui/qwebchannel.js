/****************************************************************************
**
** Copyright (C) 2017 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the QtWebChannel module of the Qt Toolkit.
**
****************************************************************************/

"use strict";

var QWebChannelMessageTypes = {
    signal: 1,
    propertyUpdate: 2,
    init: 3,
    idle: 4,
    debug: 5,
    invokeMethod: 6,
    connectToSignal: 7,
    disconnectFromSignal: 8,
    setProperty: 9,
    response: 10
};

var QWebChannel = function(transport, initCallback)
{
    if (typeof transport !== "object" || typeof transport.send !== "function") {
        console.error("The QWebChannel transport object is invalid!");
        return;
    }

    var channel = this;
    this.transport = transport;

    this.send = function(data)
    {
        if (typeof data !== "string") {
            data = JSON.stringify(data);
        }
        channel.transport.send(data);
    }

    this.transport.onmessage = function(message)
    {
        var data = message.data;
        if (typeof data === "string") {
            data = JSON.parse(data);
        }
        switch (data.type) {
            case QWebChannelMessageTypes.signal:
                channel.handleSignal(data);
                break;
            case QWebChannelMessageTypes.response:
                channel.handleResponse(data);
                break;
            case QWebChannelMessageTypes.propertyUpdate:
                channel.handlePropertyUpdate(data);
                break;
            default:
                console.error("invalid message type received: ", data.type);
                break;
        }
    }

    this.execCallbacks = {};
    this.execId = 0;
    this.exec = function(data, callback)
    {
        if (!callback) {
            channel.send(data);
            return;
        }
        var id = channel.execId++;
        channel.execCallbacks[id] = callback;
        data.id = id;
        channel.send(data);
    };

    this.handleResponse = function(data)
    {
        if (!data.hasOwnProperty("id")) {
            console.error("undefined id of response: ", data);
            return;
        }
        var res = data.res;
        if (channel.execCallbacks.hasOwnProperty(data.id)) {
            channel.execCallbacks[data.id](res);
            delete channel.execCallbacks[data.id];
        }
    };

    this.idToSignalHandlers = {};
    this.handleSignal = function(data)
    {
        var objectName = data.object;
        if (channel.objects.hasOwnProperty(objectName)) {
            var object = channel.objects[objectName];
            var signalName = data.signal;
            if (object.signals.hasOwnProperty(signalName)) {
                object.signals[signalName].apply(object, data.args);
            } else {
                console.warn("Unhandled signal: " + objectName + "::" + signalName);
            }
        } else {
            console.warn("Unhandled signal of unknown object: " + objectName);
        }
    };

    this.handlePropertyUpdate = function(data)
    {
        for (var i in data.signals) {
            var signal = data.signals[i];
            channel.handleSignal(signal);
        }
    };

    this.objects = {};

    this.exec({type: QWebChannelMessageTypes.init}, function(data) {
        for (var objectName in data) {
            var object = new QObject(objectName, data[objectName], channel);
        }
        // now setup properties, which may depend on other objects
        for (var objectName in data) {
            var object = channel.objects[objectName];
            object.__setupPropertyHandlers__(data[objectName]);
        }

        if (initCallback) {
            initCallback(channel);
        }
    });
};

function QObject(name, data, webChannel)
{
    this.__id__ = name;
    webChannel.objects[name] = this;

    // List of signals that can be connected to
    this.signals = {};

    // Internal setup of methods and signals
    this.__setupPropertyHandlers__ = function(data) {
        for (var i in data.properties) {
            var property = data.properties[i];
            var propertyName = property[0];
            var propertyValue = property[1];
            var notifySignalName = property[2];

            // set the property value
            this[propertyName] = propertyValue;

            if (notifySignalName) {
                var signal = new QSignal(this, notifySignalName, webChannel);
                this.signals[notifySignalName] = signal;
                // keep the property value up to date
                (function(obj, propName) {
                    signal.connect(function(newValue) {
                        obj[propName] = newValue;
                    });
                })(this, propertyName);
            }
        }
    }

    for (var i in data.methods) {
        var method = data.methods[i];
        var methodName = method[0];
        var methodId = method[1];

        this[methodName] = (function (obj, name, id) {
            return function () {
                var args = [];
                var callback;
                for (var i = 0; i < arguments.length; ++i) {
                    if (typeof arguments[i] === "function")
                        callback = arguments[i];
                    else
                        args.push(arguments[i]);
                }

                obj.__webChannel__.exec({
                    type: QWebChannelMessageTypes.invokeMethod,
                    object: obj.__id__,
                    method: id,
                    args: args
                }, callback);
            };
        })(this, methodName, methodId);
    }

    for (var i in data.signals) {
        var signal = data.signals[i];
        var signalName = signal[0];
        var signalId = signal[1];
        this.signals[signalName] = new QSignal(this, signalName, webChannel);
    }

    this.__webChannel__ = webChannel;
}

function QSignal(object, name, webChannel)
{
    this.handlers = [];

    this.connect = function(handler) {
        if (typeof(handler) !== "function") {
            console.error("Connect failed: handler is not a function.");
            return;
        }

        if (this.handlers.indexOf(handler) === -1) {
            this.handlers.push(handler);
            if (this.handlers.length === 1) {
                webChannel.exec({
                    type: QWebChannelMessageTypes.connectToSignal,
                    object: object.__id__,
                    signal: name
                });
            }
        }
    };

    this.disconnect = function(handler) {
        if (typeof(handler) !== "function") {
            console.error("Disconnect failed: handler is not a function.");
            return;
        }

        var index = this.handlers.indexOf(handler);
        if (index !== -1) {
            this.handlers.splice(index, 1);
            if (this.handlers.length === 0) {
                webChannel.exec({
                    type: QWebChannelMessageTypes.disconnectFromSignal,
                    object: object.__id__,
                    signal: name
                });
            }
        }
    };

    this.apply = function(object, args) {
        for (var i = 0; i < this.handlers.length; ++i) {
            this.handlers[i].apply(object, args);
        }
    };
}

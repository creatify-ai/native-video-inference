//! A single owned result, consumed by a blocking legacy adapter or a nonblocking
//! declared callback. Publishing never holds the result lock while calling work.
use std::sync::{Arc, Condvar, Mutex};

type Result<T> = std::result::Result<T, String>;
type Subscriber<T> = Box<dyn FnOnce(Result<T>) + Send>;
struct State<T> { value: Option<Result<T>>, subscriber: Option<Subscriber<T>> }
struct Shared<T> { state: Mutex<State<T>>, changed: Condvar }
pub struct ResultSender<T> { shared: Option<Arc<Shared<T>>> }
pub struct ResultReceiver<T> { shared: Arc<Shared<T>> }

pub fn channel<T>() -> (ResultSender<T>, ResultReceiver<T>) {
    let shared=Arc::new(Shared {state:Mutex::new(State {value:None,subscriber:None}),changed:Condvar::new()});
    (ResultSender {shared:Some(shared.clone())},ResultReceiver {shared})
}
impl<T> ResultSender<T> {
    fn publish(&mut self, value:Result<T>) {
        let Some(shared)=self.shared.take() else {return;};
        let subscriber={
            let mut state=shared.state.lock().unwrap();
            if let Some(subscriber)=state.subscriber.take() {Some(subscriber)}
            else {state.value=Some(value);shared.changed.notify_all();return;}
        };
        if let Some(subscriber)=subscriber {subscriber(value);}
    }
    pub fn send(mut self, value:Result<T>) {self.publish(value);}
}
impl<T> Drop for ResultSender<T> {
    fn drop(&mut self) {self.publish(Err("startup result producer exited without a result".into()));}
}
impl<T> ResultReceiver<T> {
    /// The callback may run immediately, or on the publishing thread. It must
    /// not assume an executor worker or a particular CUDA owner.
    pub fn subscribe(self, subscriber:impl FnOnce(Result<T>)+Send+'static) {
        let value={
            let mut state=self.shared.state.lock().unwrap();
            if let Some(value)=state.value.take() {value}
            else {state.subscriber=Some(Box::new(subscriber));return;}
        };
        subscriber(value);
    }
    pub fn wait(self)->Result<T> {
        let mut state=self.shared.state.lock().unwrap();
        loop {
            if let Some(value)=state.value.take() {return value;}
            state=self.shared.changed.wait(state).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_before_and_after_subscription_delivers_once() {
        for early in [false,true] {
            let (sender,receiver)=channel();let (reply,answer)=std::sync::mpsc::channel();
            let mut sender=Some(sender);
            if early {sender.take().unwrap().send(Ok(7));}
            receiver.subscribe(move |result|reply.send(result).unwrap());
            if let Some(sender)=sender {sender.send(Ok(7));}
            assert_eq!(answer.recv().unwrap(),Ok(7));assert!(answer.try_recv().is_err());
        }
    }
    #[test]
    fn producer_failure_wakes_both_adapters() {
        let (sender,receiver)=channel::<()>();drop(sender);assert!(receiver.wait().is_err());
        let (sender,receiver)=channel::<()>();let (reply,answer)=std::sync::mpsc::channel();
        receiver.subscribe(move |result|reply.send(result).unwrap());drop(sender);
        assert!(answer.recv().unwrap().is_err());
    }
}
